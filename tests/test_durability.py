"""Big results spilled to files, compaction that survives a reopen, audit log, rendering."""

import io
import json
import re

from rich.console import Console

from conftest import call, turn
from minicw.core.permissions import deny_all
from minicw.core.sessions import Session
from minicw.ui.render import Renderer


def test_a_huge_result_is_spilled_and_can_be_read_back(make, ws, state):
    (ws / "big.txt").write_text("\n".join(f"line {i} " + "x" * 190 for i in range(300)))
    engine, _p, session, _e = make([
        turn(calls=[call("grep", "g1", pattern="line", path="big.txt", max_results=200)]),
        turn("done"),
    ])
    engine.run("search")
    text = [m for m in engine.messages if m["role"] == "tool"][0]["content"]
    assert len(text) < 12_000 and "characters omitted" in text
    path = re.search(r"saved at (.+?) : open it", text).group(1)
    assert session.id in path and "spill" in path
    # the model can open it with read_file, because the spill folder is a read-only root...
    again, _p2, _s2, _e2 = make([turn(calls=[call("read_file", "r1", path=path, start_line=1, max_lines=3)]),
                                 turn("ok")], session=session, messages=engine.messages)
    again.run("open it")
    assert "line 0" in [m for m in again.messages if m["role"] == "tool"][-1]["content"]
    # ...and cannot write into it
    third, _p3, _s3, _e3 = make([turn(calls=[call("write_file", "w1", path=path, content="tampered")]), turn("ok")],
                                session=session, messages=again.messages, mode="bypass")
    third.run("tamper")
    assert "outside" in [m for m in third.messages if m["role"] == "tool"][-1]["content"]


def test_compaction_is_stored_with_the_session_and_survives_a_reopen(make, state):
    history = []
    for i in range(12):
        history += [{"role": "user", "content": f"question {i} " + "q" * 300},
                    {"role": "assistant", "content": f"answer {i} " + "a" * 300}]
    summary = "## Primary request\nthe user asked many things. " + "details " * 40
    session = Session.create(state, workspace="w", backend="ollama", model="fake", mode="auto")
    for message in history:  # an earlier part of the conversation, already on disk
        session.append(message)
    engine, provider, _s, events = make([turn(summary), turn("final answer")], session=session,
                                        messages=list(history), context_window=1000)
    length_before = len(engine.messages)
    assert engine.run("one more question").status == "completed"
    assert engine.compaction_state is not None and not engine.compaction_state.trimmed
    assert any(e["type"] == "compacted" for e in events)
    # the transcript only grew; the sent request carries the summary instead of the old turns
    assert len(engine.messages) == length_before + 2
    sent = provider.calls[1]["messages"]
    assert any("compacted" in str(m.get("content")) and "Primary request" in str(m.get("content")) for m in sent)
    assert len(sent) < len(engine.messages)
    # reopen: the compacted view is restored
    reopened, messages = Session.open(state, session.id)
    engine2, provider2, _s, _e = make([turn("later")], session=reopened, messages=messages, context_window=1000)
    assert engine2.compaction_state is not None
    assert engine2.compaction_state.boundary == engine.compaction_state.boundary
    engine2.run("and then?")
    assert len(provider2.calls[0]["messages"]) < len(engine2.messages)


def test_the_audit_log_records_who_decided_what(make, state):
    engine, _p, session, _e = make([
        turn(calls=[call("write_file", "w1", path="a.txt", content="x"),
                    call("run_shell", "s1", command="rm -rf /")]),
        turn("ok"),
    ], mode="auto")
    engine.run("go")
    lines = [json.loads(l) for l in (state / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    by_tool = {l["tool"]: l for l in lines}
    assert by_tool["write_file"]["verdict"] == "allow" and by_tool["write_file"]["risk"] == "write_local"
    assert by_tool["write_file"]["source"] == "mode" and by_tool["write_file"]["outcome"] == "ok"
    assert by_tool["run_shell"]["verdict"] == "deny" and by_tool["run_shell"]["source"] == "floor"
    assert all(l["session"] == session.id for l in lines)


def test_denied_by_the_user_is_audited_as_such(make, state):
    engine, _p, _s, _e = make([turn(calls=[call("write_file", "w1", path="a.txt", content="x")]), turn("ok")],
                              mode="ask", approver=deny_all)
    engine.run("go")
    entry = json.loads((state / "audit.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert entry["verdict"] == "ask" and entry["source"] == "user" and entry["outcome"] == "denied"


def test_model_text_that_looks_like_markup_does_not_break_the_stream():
    out = io.StringIO()
    renderer = Renderer(Console(file=out, width=100, force_terminal=False))
    renderer.on_event({"type": "assistant_delta", "text": "closing tag [/bold] and [red]colour[/red]"})
    renderer.on_event({"type": "assistant_message", "text": ""})
    renderer.on_event({"type": "tool_proposed", "name": "grep", "arguments": {"pattern": "[/x]"}})
    renderer.on_event({"type": "tool_finished", "name": "grep", "status": "ok", "preview": "[bold]"})
    renderer.on_event({"type": "interrupted"})
    shown = out.getvalue()
    assert "[/bold]" in shown and "[red]colour[/red]" in shown and "[/x]" in shown and "interrupted" in shown


def test_command_line_overrides_and_continue(state, ws):
    from minicw.cli import apply_overrides, build_parser, open_or_create_session
    from minicw.core.config import Settings

    args = build_parser("chat").parse_args(["--backend", "ollama", "--mode", "plan", "--continue"])
    settings = apply_overrides(Settings(), args).resolved()
    assert (settings.backend, settings.mode, settings.model) == ("ollama", "plan", "mcw-qwen3.5-4b")
    first, _ = open_or_create_session(args, settings, state, ws)  # nothing to continue yet: a new session
    first.append({"role": "user", "content": "hello"})
    again, messages = open_or_create_session(args, settings, state, ws)
    assert again.id == first.id and [m["content"] for m in messages] == ["hello"]
