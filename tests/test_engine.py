"""Multi-turn behaviour of the engine, tool calls, denial, plan mode."""

import json

from conftest import call, turn
from minicw.core.permissions import Answer, deny_all
from minicw.core.sessions import Session


def roles(messages):
    return [m["role"] for m in messages]


def test_two_turns_share_one_history(make):
    engine, provider, _s, _e = make([turn("hello"), turn("again")])
    assert engine.run("first").status == "completed"
    assert engine.run("second").status == "completed"
    sent = provider.calls[1]["messages"]
    assert roles(sent) == ["system", "user", "assistant", "user"]
    assert sent[1]["content"] == "first" and sent[3]["content"] == "second"
    assert roles(engine.messages) == ["user", "assistant", "user", "assistant"]


def test_tool_call_gets_exactly_one_result_with_matching_id(make, ws):
    engine, provider, _s, events = make([
        turn(calls=[call("write_file", "w1", path="a.txt", content="hi")]),
        turn("done"),
    ])
    result = engine.run("make a file")
    assert result.status == "completed"
    assert (ws / "a.txt").read_text() == "hi"
    tool_msgs = [m for m in engine.messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["w1"]
    assert engine.pending_calls() == []
    assert [e["type"] for e in events if e["type"].startswith("tool_")] == ["tool_proposed", "tool_finished"]


def test_denied_call_still_gets_a_result(make, ws):
    engine, _p, _s, _e = make([turn(calls=[call("write_file", "w1", path="a.txt", content="x")]), turn("ok")],
                              mode="ask", approver=deny_all)
    engine.run("write")
    assert not (ws / "a.txt").exists()
    tool = [m for m in engine.messages if m["role"] == "tool"][0]
    assert tool["status"] == "denied" and "declined" in tool["content"]


def test_write_outside_workspace_is_denied_even_in_bypass(make, ws):
    engine, _p, _s, _e = make([turn(calls=[call("write_file", "w1", path="../evil.txt", content="x")]), turn("ok")],
                              mode="bypass")
    engine.run("write")
    assert not (ws.parent / "evil.txt").exists()
    assert "outside" in [m for m in engine.messages if m["role"] == "tool"][0]["content"]


def test_always_answer_is_remembered_and_saved(make, state, ws):
    asked = []

    def approver(tool, args, decision):
        asked.append(tool)
        return Answer.ALWAYS

    cmd = "echo hello > out.txt"
    engine, _p, session, _e = make([
        turn(calls=[call("run_shell", "s1", command=cmd)]),
        turn(calls=[call("run_shell", "s2", command=cmd)]),
        turn("done"),
    ], mode="ask", approver=approver)
    engine.run("go")
    assert asked == ["run_shell"]  # the second identical command did not ask again
    reopened, _msgs = Session.open(state, session.id)
    assert reopened.meta["grants"]["commands"] == [cmd]


def test_unknown_tool_and_bad_arguments_are_reported_not_raised(make):
    engine, _p, _s, _e = make([
        turn(calls=[call("nope", "n1"), call("read_file", "n2", wrong=1)]),
        turn("ok"),
    ])
    assert engine.run("x").status == "completed"
    results = {m["tool_call_id"]: m["content"] for m in engine.messages if m["role"] == "tool"}
    assert "unknown tool" in results["n1"] and "bad arguments" in results["n2"]


def test_plan_mode_blocks_writes_then_plan_approval_switches_mode(make, ws):
    approved = []
    engine, provider, _s, events = make([
        turn(calls=[call("write_file", "w1", path="a.txt", content="x")]),
        turn(calls=[call("propose_plan", "p1", plan="1. write a.txt")]),
        turn("plan accepted"),
    ], mode="plan", plan_approver=lambda plan: (approved.append(plan) or (True, "")))
    engine.run("do it")
    assert not (ws / "a.txt").exists()
    assert approved == ["1. write a.txt"]
    assert engine.permissions.mode.value == "ask"
    assert {"type": "mode_changed", "mode": "ask"} in events
    # the plan notice is in what was sent, but is not stored in the history
    assert "[Plan mode]" in provider.calls[0]["messages"][-1]["content"]
    assert "[Plan mode]" not in engine.messages[0]["content"]


def test_ask_user_returns_the_answer(make):
    engine, _p, _s, _e = make([turn(calls=[call("ask_user", "q1", question="which?", options=["a", "b"])]),
                               turn("thanks")],
                              ask_user=lambda q, opts: opts[1])
    engine.run("go")
    tool = [m for m in engine.messages if m["role"] == "tool"][0]
    assert json.loads(tool["content"]) == {"answer": "b"}


def test_cut_off_reply_is_continued(make):
    engine, provider, _s, _e = make([turn("part one", finish="length"), turn("part two")])
    result = engine.run("write")
    assert result.status == "completed" and result.text == "part two"
    assert "cut off" in provider.calls[1]["messages"][-1]["content"]
