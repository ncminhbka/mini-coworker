"""Session files: append-only transcript, safe ids, torn lines, redaction, reopening."""

import pytest

from conftest import turn
from minicw.core.compaction import CompactionState
from minicw.core.sessions import Session, is_safe_session_id


def test_unsafe_ids_are_refused(state):
    for bad in ("../x", "..\\x", "a/b", "", "x" * 200, "a.b"):
        assert not is_safe_session_id(bad)
        with pytest.raises(ValueError):
            Session.open(state, bad)
    with pytest.raises(ValueError):
        Session.create(state, workspace="w", backend="b", model="m", mode="ask", session_id="../evil")
    assert not list(state.glob("**/evil*"))


def test_transcript_is_append_only_and_roundtrips(state):
    s = Session.create(state, workspace="w", backend="b", model="m", mode="ask")
    s.append({"role": "user", "content": "one"})
    s.append({"role": "assistant", "content": "two"})
    before = s.path_jsonl.read_bytes()
    s.append({"role": "user", "content": "three"})
    assert s.path_jsonl.read_bytes().startswith(before)
    _reopened, messages = Session.open(state, s.id)
    assert [m["content"] for m in messages] == ["one", "two", "three"]
    assert s.meta["title"] == "one"  # the title comes from the first user message


def test_torn_last_line_is_dropped_and_the_file_repaired(state):
    s = Session.create(state, workspace="w", backend="b", model="m", mode="ask")
    s.append({"role": "user", "content": "ok"})
    with open(s.path_jsonl, "ab") as fh:
        fh.write(b'{"role": "assistant", "content": "cut of')  # the process died mid-write
    reopened, messages = Session.open(state, s.id)
    assert [m["content"] for m in messages] == ["ok"]
    assert reopened.path_jsonl.read_bytes().endswith(b"\n")
    reopened.append({"role": "user", "content": "after"})
    _s, again = Session.open(state, s.id)
    assert [m["content"] for m in again] == ["ok", "after"]


# Built at run time so that no key-shaped string sits in the source (GitHub push protection flags those).
FAKE_KEY = "sk-" + "abcdefghijklmnop" + "qrstuvwx"


def test_secrets_never_reach_the_files(state, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
    s = Session.create(state, workspace="w", backend="b", model="m", mode="ask")
    s.append({"role": "tool", "content": "the key is " + FAKE_KEY})
    s.save_meta(note=FAKE_KEY)
    for path in (s.path_jsonl, s.path_meta):
        assert "abcdefghijklmnop" not in path.read_text(encoding="utf-8")


def test_list_is_newest_first_and_filtered_by_workspace(state):
    a = Session.create(state, workspace="one", backend="b", model="m", mode="ask")
    b = Session.create(state, workspace="two", backend="b", model="m", mode="ask")
    a.save_meta(title="later")
    assert [m["id"] for m in Session.list(state)] == [a.id, b.id]
    assert [m["id"] for m in Session.list(state, "two")] == [b.id]


def test_reopening_restores_mode_grants_usage_and_history(make, state):
    engine, _p, session, _e = make([turn("hi")], mode="ask")
    engine.permissions.load_grants({"tools": ["web_fetch"], "commands": ["git status"]})
    engine.run("hello")
    reopened, messages = Session.open(state, session.id)
    engine2, _p2, _s2, _e2 = make([turn("again")], session=reopened, messages=messages, mode="plan")
    assert engine2.permissions.mode.value == "ask"  # the session's mode wins over the argument
    assert engine2.permissions.grants() == {"tools": ["web_fetch"], "commands": ["git status"]}
    assert engine2.usage["turns"] == 1
    assert [m["role"] for m in engine2.messages] == ["user", "assistant"]


def test_a_saved_compaction_that_does_not_fit_the_history_is_dropped(make, state):
    s = Session.create(state, workspace="w", backend="ollama", model="fake", mode="ask")
    s.save_meta(compaction=CompactionState(boundary=99, summary="x", working_state="", user_messages=[]).as_dict())
    engine, _p, _s, _e = make([turn("ok")], session=s, messages=[{"role": "user", "content": "a"}])
    assert engine.compaction_state is None
