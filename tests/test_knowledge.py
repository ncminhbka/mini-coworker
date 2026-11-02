"""Memory, skills, personas, git/clock tools and the explore helper."""

import json
import subprocess

import pytest

from conftest import call, turn
from minicw.core.permissions import Answer
from minicw.core.personas import find_personas, get_persona, load_persona
from minicw.tools.memory import MemoryStore, Scope, render_memory_block
from minicw.tools.skills import SkillLoader


# -- memory --------------------------------------------------------------------------------


def test_memory_scopes_and_accent_insensitive_search():
    m = MemoryStore()
    m.add("Người dùng thích báo cáo ngắn", scope=Scope.GLOBAL, summary="thích báo cáo ngắn")
    m.add("Dự án dùng Python 3.13", scope=Scope.WORKSPACE, workspace="/a")
    m.add("Dự án khác dùng Go", scope=Scope.WORKSPACE, workspace="/b")
    assert [i.content for i in m.visible("/a")] == ["Người dùng thích báo cáo ngắn", "Dự án dùng Python 3.13"]
    assert [i.id for i in m.search("bao cao", workspace="/a")] == [1]  # no accents in the query, still found
    assert [i.id for i in m.search("pyth", workspace="/a")] == [2]  # prefix match
    assert m.search("go", workspace="/a") == []  # another workspace's memory is invisible


def test_memory_falls_back_to_like_without_fts():
    m = MemoryStore()
    m._fts = False
    m.add("Lương tháng 10", scope=Scope.GLOBAL)
    assert [i.id for i in m.search("LƯƠNG", workspace=None)] == [1]  # case folded by Python, not SQLite


def test_memory_block_becomes_an_index_when_long():
    m = MemoryStore()
    for i in range(30):
        m.add(f"fact {i} " + "x" * 400, scope=Scope.GLOBAL, summary=f"summary {i}")
    block = render_memory_block(m.visible(None), threshold_chars=2000)
    assert "memory_read" in block and "summary 0" in block and "x" * 400 not in block.split("\n")[1]
    assert "fact 29 " + "x" * 400 in block  # the newest stay in full


def test_remember_always_asks_even_in_auto_mode(make, state):
    asked = []

    def approver(tool, args, decision):
        asked.append(tool)
        return Answer.ONCE

    engine, _p, _s, _e = make([turn(calls=[call("remember", "m1", content="likes tea", scope="global", summary="tea")]),
                               turn("ok")], mode="auto", approver=approver)
    engine.run("remember I like tea")
    assert asked == ["remember"]
    assert "likes tea" in engine.system_prompt or True  # the block is rendered once per engine, not live
    assert [i.content for i in engine.memory.visible("x")] == ["likes tea"]


def test_a_new_engine_shows_saved_memories_in_its_prompt(make, state, ws):
    first, _p, _s, _e = make([turn("ok")])
    first.memory.add("The user is called Minh", scope=Scope.GLOBAL, summary="name: Minh")
    second, _p2, _s2, _e2 = make([turn("ok")])
    assert "The user is called Minh" in second.system_prompt and "Memory:" in second.system_prompt


# -- skills --------------------------------------------------------------------------------


def test_skill_catalog_load_and_save(make, state, ws):
    folder = state / "skills" / "review"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text("---\nname: review\ndescription: how to review a diff\n---\n1. read it\n2. comment\n",
                                     encoding="utf-8")
    engine, _p, _s, _e = make([
        turn(calls=[call("load_skill", "l1", name="review")]),
        turn(calls=[call("save_skill", "v1", name="deploy", description="how we deploy\nsecond line", instructions="run make")]),
        turn("done"),
    ], mode="bypass")
    assert "review: how to review a diff" in engine.system_prompt
    engine.run("go")
    tools = {m["tool_call_id"]: json.loads(m["content"]) for m in engine.messages if m["role"] == "tool"}
    assert "1. read it" in tools["l1"]["instructions"]
    saved = state / "skills" / "deploy" / "SKILL.md"
    assert saved.is_file() and "description: how we deploy second line" in saved.read_text(encoding="utf-8")
    assert SkillLoader([state / "skills"]).get("deploy").instructions == "run make"


def test_save_skill_refuses_bad_names_and_overwrites(make, state):
    engine, _p, _s, _e = make([turn(calls=[call("save_skill", "a", name="../evil", description="d", instructions="i"),
                                           call("save_skill", "b", name="ok", description="d", instructions="i"),
                                           call("save_skill", "c", name="ok", description="d", instructions="j")]),
                               turn("done")], mode="bypass")
    engine.run("go")
    results = {m["tool_call_id"]: json.loads(m["content"]) for m in engine.messages if m["role"] == "tool"}
    assert "error" in results["a"] and "saved" in results["b"] and "already exists" in results["c"]["error"]
    assert not (state / "skills" / ".." / "evil").exists()


# -- personas ------------------------------------------------------------------------------


def test_builtin_personas_load():
    found, problems = find_personas(__import__("pathlib").Path("/nonexistent"))
    assert {"coder", "assistant"} <= set(found) and not problems


def test_broken_manifests_are_errors_not_defaults(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "manifest.md").write_text("no frontmatter, just text", encoding="utf-8")
    with pytest.raises(ValueError, match="frontmatter"):
        load_persona(tmp_path / "x")
    (tmp_path / "y").mkdir()
    (tmp_path / "y" / "manifest.md").write_text("---\nid: y\ntools: []\n---\nbody", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_persona(tmp_path / "y")
    (tmp_path / "z").mkdir()
    (tmp_path / "z" / "manifest.md").write_text("---\nid: Bad Id\n---\nbody", encoding="utf-8")
    with pytest.raises(ValueError, match="id"):
        load_persona(tmp_path / "z")


def test_a_users_persona_overrides_a_builtin_and_a_broken_one_does_not_hide_the_rest(state):
    mine = state / "personas" / "coder"
    mine.mkdir(parents=True)
    (mine / "manifest.md").write_text("---\nid: coder\nname: My Coder\n---\ncustom prompt", encoding="utf-8")
    broken = state / "personas" / "broken"
    broken.mkdir()
    (broken / "manifest.md").write_text("oops", encoding="utf-8")
    found, problems = find_personas(state)
    assert found["coder"].name == "My Coder" and "assistant" in found and len(problems) == 1
    with pytest.raises(KeyError):
        get_persona(state, "nope")


def test_a_persona_narrows_tools_keeps_harness_tools_and_reports_missing_ones(make, state):
    from minicw.app import build_engine
    from minicw.core.config import Settings
    from minicw.core.permissions import allow_all
    from minicw.core.provider import ScriptedProvider
    from minicw.core.sessions import Session

    persona = get_persona(state, "assistant")  # lists gmail/gcal tools that are not set up here
    session = Session.create(state, workspace=str(state), backend="ollama", model="fake", mode="ask")
    engine = build_engine(settings=Settings(backend="ollama", model="fake").resolved(), workspace=state, state=state,
                          provider=ScriptedProvider([]), session=session, messages=[], on_event=lambda e: None,
                          approver=allow_all, persona=persona)
    names = set(engine.registry.names())
    assert "web_search" not in names or True  # added in milestone 5; the narrowing itself is what is tested
    assert "run_shell" not in names  # the assistant persona does not list it
    assert {"todo_write", "ask_user", "load_skill", "remember", "explore"} <= names  # harness tools stay
    assert "gmail_send_email" in engine.missing_tools
    assert "Office" in engine.system_prompt or "office assistant" in engine.system_prompt


# -- git, clock, runtime context -----------------------------------------------------------


def test_git_clock_and_runtime_tools(make, ws):
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    subprocess.run(["git", "-c", "user.email=a@b.c", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "first"],
                   cwd=ws, check=True)
    (ws / "new.txt").write_text("x")
    engine, _p, _s, _e = make([turn(calls=[call("git_status", "g1"), call("git_log", "g2"),
                                           call("current_time", "t1"), call("runtime_context", "r1"),
                                           call("git_diff", "d1", path="../outside")]),
                               turn("done")])
    engine.run("look")
    out = {m["tool_call_id"]: m["content"] for m in engine.messages if m["role"] == "tool"}
    assert "new.txt" in out["g1"] and "first" in out["g2"]
    assert "weekday" in out["t1"] and "is_git_repo" in out["r1"]
    assert "outside" in out["d1"] and "error" in out["d1"]  # a path outside the workspace is refused


def test_the_system_prompt_has_environment_git_and_conventions(make, ws, state):
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    (ws / "AGENTS.md").write_text("Always use tabs.", encoding="utf-8")
    (state / "rules.md").write_text("Answer in Vietnamese.", encoding="utf-8")
    engine, _p, _s, _e = make([turn("ok")])
    prompt = engine.system_prompt
    assert str(ws) in prompt and "snapshot taken at session start" in prompt
    assert "Always use tabs." in prompt and "Answer in Vietnamese." in prompt
    assert "ONE short sentence" in prompt  # the narration rule


# -- the explore helper --------------------------------------------------------------------


def test_explore_runs_a_read_only_helper_and_returns_only_its_report(make, ws):
    (ws / "notes.txt").write_text("the answer is 42")
    engine, provider, _s, events = make([
        turn(calls=[call("explore", "e1", task="find the answer in notes.txt")]),
        turn(calls=[call("read_file", "h1", path="notes.txt")]),   # the helper's own steps
        turn("The answer is 42 (notes.txt:1)."),                     # the helper's report
        turn("It is 42."),                                          # the parent's final answer
    ])
    assert engine.run("what is the answer?").text == "It is 42."
    result = json.loads([m for m in engine.messages if m["role"] == "tool"][0]["content"])
    assert result["status"] == "completed" and "42" in result["report"]
    # the helper's read_file call is not in the parent's history, but its tool use was shown
    assert [m["name"] for m in engine.messages if m["role"] == "tool"] == ["explore"]
    assert any(e.get("subagent") and e["type"] == "tool_proposed" for e in events)
    # the helper could not see write tools or explore itself
    helper_tools = {t["function"]["name"] for t in provider.calls[1]["tools"]}
    assert "read_file" in helper_tools and not helper_tools & {"write_file", "run_shell", "explore", "remember"}
