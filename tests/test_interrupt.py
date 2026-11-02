"""Ctrl-C: real SIGINT delivered to this process while a turn is in progress."""

import os
import signal
import subprocess
import sys
import threading
import time

from conftest import call, turn
from minicw.core.provider import Hang


def sigint_after(seconds: float) -> threading.Timer:
    timer = threading.Timer(seconds, signal.raise_signal, [signal.SIGINT])
    timer.start()
    return timer


def pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def test_ctrl_c_during_stream_keeps_partial_text_and_the_engine_usable(make):
    engine, provider, session, events = make([Hang("I was saying"), turn("back again")])
    timer = sigint_after(0.3)
    started = time.monotonic()
    result = engine.run("talk")
    timer.cancel()
    assert result.status == "interrupted"
    assert time.monotonic() - started < 3
    assert engine.messages[-1]["role"] == "assistant" and engine.messages[-1]["content"] == "I was saying"
    assert any(e["type"] == "interrupted" for e in events)
    # the very next turn works, and the interrupted text is part of the history it sees
    assert engine.run("continue").status == "completed"
    assert provider.calls[1]["messages"][-2]["content"] == "I was saying"


def test_ctrl_c_during_a_shell_command_kills_the_process_tree(make, ws):
    pidfile = ws / "pid.txt"
    code = f"import os,time; open(r'{pidfile}','w').write(str(os.getpid())); time.sleep(60)"
    if sys.platform == "win32":
        command = f'& "{sys.executable}" -c "{code}"'
    else:
        command = f"'{sys.executable}' -c \"{code}\""
    engine, _p, _s, _e = make([turn(calls=[call("run_shell", "s1", command=command, timeout_seconds=120)]),
                               turn("never reached")])
    timer = sigint_after(2.5)
    started = time.monotonic()
    result = engine.run("sleep")
    timer.cancel()
    assert result.status == "interrupted"
    assert time.monotonic() - started < 15
    tool = [m for m in engine.messages if m["role"] == "tool"]
    assert len(tool) == 1 and tool[0]["tool_call_id"] == "s1"
    assert engine.pending_calls() == []
    pid = int(pidfile.read_text())
    time.sleep(0.5)
    assert not pid_alive(pid), "the command's process survived the interrupt"


def test_interrupt_closes_the_rest_of_a_batch(make, ws):
    command = "Start-Sleep 30" if sys.platform == "win32" else "sleep 30"
    engine, _p, _s, _e = make([turn(calls=[call("run_shell", "s1", command=command),
                                           call("write_file", "w1", path="a.txt", content="x")]),
                               turn("never")])
    timer = sigint_after(1.5)
    result = engine.run("go")
    timer.cancel()
    assert result.status == "interrupted"
    ids = [m["tool_call_id"] for m in engine.messages if m["role"] == "tool"]
    assert ids == ["s1", "w1"]  # both answered, in order
    assert not (ws / "a.txt").exists()  # the call after the interrupt was NOT run


def test_ctrl_c_at_the_permission_prompt_means_no_and_stops_the_turn(make, ws):
    def approver(tool, args, decision):
        raise KeyboardInterrupt  # what a prompt_toolkit prompt raises on Ctrl-C

    engine, _p, _s, _e = make([turn(calls=[call("write_file", "w1", path="a.txt", content="x"),
                                           call("write_file", "w2", path="b.txt", content="y")]),
                               turn("never")], mode="ask", approver=approver)
    result = engine.run("write")
    assert result.status == "interrupted"
    assert not (ws / "a.txt").exists() and not (ws / "b.txt").exists()
    assert engine.pending_calls() == []


def test_orphans_from_a_crash_are_closed_at_the_start_of_the_next_turn(make):
    orphan = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "x1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]}
    engine, provider, _s, _e = make([turn("ok")], messages=[{"role": "user", "content": "old"}, orphan])
    engine.run("new")
    sent = provider.calls[0]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool", "user"]
    assert sent[3]["tool_call_id"] == "x1"
