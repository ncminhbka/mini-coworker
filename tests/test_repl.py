"""The interactive loop, driven by scripted keyboard input and a scripted model."""

import io

import pytest
from rich.console import Console

from conftest import call, turn
from minicw.core.config import Settings
from minicw.core.provider import ScriptedProvider
from minicw.core.sessions import Session
from minicw.ui import repl as repl_module
from minicw.ui.repl import PlainPrompter, Repl


@pytest.fixture
def start(state, ws, monkeypatch):
    """start(turns, keyboard) -> (repl, screen text getter). `keyboard` is the list of lines typed."""
    monkeypatch.setenv("MINICW_STATE_DIR", str(state))
    monkeypatch.setattr(repl_module, "make_prompters", lambda history: (PlainPrompter(), PlainPrompter()))

    def _start(turns, keyboard, mode="ask"):
        lines = iter(keyboard)

        def fake_input(prompt=""):
            try:
                return next(lines)
            except StopIteration:
                raise EOFError from None  # the user pressed Ctrl-D

        monkeypatch.setattr("builtins.input", fake_input)
        out = io.StringIO()
        settings = Settings(backend="ollama", model="fake", mode=mode).resolved()
        session = Session.create(state, workspace=str(ws), backend="ollama", model="fake", mode=mode)
        repl = Repl(console=Console(file=out, width=110, force_terminal=False), settings=settings, state=state,
                    workspace=ws, provider=ScriptedProvider(turns), session=session, messages=[])
        return repl, out.getvalue
    return _start


def test_a_conversation_with_two_turns_and_a_clean_exit(start):
    repl, screen = start([turn("Hello there"), turn("Second answer")], ["hi", "and again", "/exit"])
    assert repl.run() == 0
    text = screen()
    assert "Hello there" in text and "Second answer" in text
    assert [m["role"] for m in repl.engine.messages] == ["user", "assistant", "user", "assistant"]


def test_ctrl_d_leaves_and_an_empty_session_is_not_kept(start, state):
    repl, _screen = start([], [])  # EOF at once
    sid = repl.session.id
    assert repl.run() == 0
    assert not (state / "sessions" / f"{sid}.json").exists()


def test_a_session_with_messages_is_kept_and_resumable(start, state):
    repl, screen = start([turn("noted")], ["remember this", "/sessions", "/exit"])
    repl.run()
    sid = repl.session.id
    assert sid in screen() and "remember this" in screen()
    opened, messages = Session.open(state, sid)
    assert [m["content"] for m in messages] == ["remember this", "noted"]


def test_new_and_resume_switch_between_conversations(start, state):
    repl, screen = start([turn("first answer")], ["question one", "/new", "/exit"])
    repl.run()
    sessions = Session.list(state)
    assert len(sessions) == 1  # /new left the old session on disk; the fresh one was empty and was discarded
    first = sessions[0]["id"]
    repl2, screen2 = start([], [f"/resume {first}", "/exit"])
    repl2.run()
    assert [m["content"] for m in repl2.engine.messages] == ["question one", "first answer"]
    assert "2 messages" in screen2()


def test_the_permission_prompt_in_the_loop(start, ws):
    repl, screen = start([turn(calls=[call("write_file", "w1", path="a.txt", content="x")]), turn("written")],
                         ["please write", "y", "/exit"])
    repl.run()
    assert (ws / "a.txt").exists() and "permission needed" in screen()


def test_answering_no_keeps_the_file_out(start, ws):
    repl, _screen = start([turn(calls=[call("write_file", "w1", path="a.txt", content="x")]), turn("ok then")],
                          ["please write", "n", "/exit"])
    repl.run()
    assert not (ws / "a.txt").exists()


def test_always_is_remembered_across_turns(start, ws):
    repl, screen = start([turn(calls=[call("write_file", "w1", path="a.txt", content="1")]), turn("one"),
                          turn(calls=[call("write_file", "w2", path="b.txt", content="2")]), turn("two")],
                         ["first", "a", "second", "/exit"])
    repl.run()
    assert (ws / "b.txt").exists() and screen().count("permission needed") == 1


def test_ask_user_and_plan_flow_in_the_loop(start, ws):
    repl, screen = start([turn(calls=[call("ask_user", "q1", question="Which file?", options=["a.txt", "b.txt"])]),
                          turn("using b.txt")], ["start", "2", "/exit"])
    repl.run()
    assert "Which file?" in screen() and "using b.txt" in screen()
    assert "b.txt" in [m for m in repl.engine.messages if m["role"] == "tool"][0]["content"]

    repl2, screen2 = start([turn(calls=[call("propose_plan", "p1", plan="1. do it")]), turn("starting"),
                            turn(calls=[call("write_file", "w1", path="done.txt", content="x")]), turn("finished")],
                           ["/mode plan", "plan it", "y", "go on", "y", "/exit"], mode="ask")
    repl2.run()
    assert "proposed plan" in screen2() and "mode → ask" in screen2() and (ws / "done.txt").exists()


def test_slash_commands(start):
    repl, screen = start([], ["/help", "/mode auto", "/mode nonsense", "/model other-model", "/tokens", "/bogus",
                              "/persona", "/persona coder", "/persona none", "/skills", "/memory", "/mcp", "/exit"])
    repl.run()
    text = screen()
    assert "/resume" in text and "mode → auto" in text and "usage: /mode" in text
    assert "model → other-model" in text and "this session" in text and "unknown command /bogus" in text
    assert "coder" in text and "persona coder" in text and "nothing remembered yet" in text
    assert "no MCP servers configured" in text
    assert repl.persona is None and repl.engine.model == "other-model"
