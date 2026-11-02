"""Assembly: turns settings + a session into a ready Engine.

One place knows which tools exist, how the permission engine is wired, and where state goes.
The terminal UI and the headless `run` command both build their engine here. `/new`, `/resume`
and `/persona` call it again for a fresh engine, because the system prompt (persona, memory,
skills) is rendered once per engine and must not change in the middle of a conversation.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .core.audit import AuditLog
from .core.compaction import CompactionState
from .core.config import Settings
from .core.engine import Engine, EngineConfig
from .core.permissions import Approver, Mode, PermissionEngine
from .core.personas import HARNESS_TOOLS, Persona
from .core.prompt import build_system_prompt
from .core.provider import OpenAICompatProvider
from .core.redact import redact_json
from .core.sessions import Session
from .core.workspace import Workspace
from .tools.files import make_file_tools
from .tools.interact import make_interact_tools
from .tools.memory import MemoryStore, make_memory_tools, memory_prompt_block
from .tools.registry import ToolRegistry, ToolSpec
from .tools.results import ResultSpiller
from .tools.shell import make_shell_tool
from .tools.skills import SkillLoader, make_skill_tools
from .tools.subagent import make_explore_tool
from .tools.system import make_system_tools
from .tools.todo import TodoList, make_todo_tool
from .tools.web import make_web_tools


def make_provider(backend: str, state: Path, *, debug: bool = False) -> OpenAICompatProvider:
    """Raises ProviderError when the backend needs a key that is not set."""
    provider = OpenAICompatProvider.for_backend(backend)
    if debug:
        log = state / "debug.jsonl"

        def write(kind: str, data: Any) -> None:  # redacted: a key never reaches this file
            with open(log, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(redact_json({"kind": kind, "data": data}) + "\n")

        provider.debug = write
    return provider


def build_engine(
    *,
    settings: Settings,
    workspace: Path,
    state: Path,
    provider: OpenAICompatProvider,
    session: Session,
    messages: list[dict[str, Any]],
    on_event: Callable[[dict[str, Any]], None],
    approver: Approver,
    ask_user: Optional[Callable[[str, list[str]], Optional[str]]] = None,
    plan_approver: Optional[Callable[[str], tuple[bool, str]]] = None,
    persona: Optional[Persona] = None,
    extra_tools: Sequence[ToolSpec] = (),
) -> Engine:
    cancel = threading.Event()  # set by Ctrl-C during a turn; the shell tool and the engine both watch it
    ws = Workspace(workspace)
    registry = ToolRegistry()
    for spec in make_file_tools(ws):
        registry.register(spec)
    registry.register(make_shell_tool(ws, cancel))
    registry.register(make_todo_tool(TodoList()))
    for spec in make_interact_tools() + make_system_tools(ws) + make_web_tools():
        registry.register(spec)

    memory = MemoryStore(state / "memory.db")
    for spec in make_memory_tools(memory, workspace=str(ws.root)):
        registry.register(spec)

    skill_dirs = [state / "skills", ws.root / ".minicw" / "skills"]
    if persona is not None and persona.skills_dir is not None:
        skill_dirs.append(persona.skills_dir)
    skills = SkillLoader(skill_dirs)
    allowed_skills = set(persona.skills) if persona is not None and persona.skills else None
    for spec in make_skill_tools(skills, save_dir=state / "skills", allowed=allowed_skills):
        registry.register(spec)

    for spec in extra_tools:  # MCP servers, Google: registered before the persona narrows the set
        registry.register(spec)
    holder: dict[str, Engine] = {}
    registry.register(make_explore_tool(lambda: holder["engine"], ws, state))

    # A persona narrows the tools to the ones it lists, plus the harness's own. A listed tool that
    # is not available (Google not set up, say) is dropped and reported, never a crash.
    missing: list[str] = []
    if persona is not None and persona.tools:
        missing = sorted(set(persona.tools) - set(registry.names()))
        registry.keep_only(set(persona.tools) | (HARNESS_TOOLS & set(registry.names())))

    mode = Mode(session.meta.get("mode") or settings.mode)
    permissions = PermissionEngine(registry, ws, mode)
    permissions.load_grants(session.meta.get("grants"))

    engine = Engine(
        provider, session.meta.get("model") or settings.model, registry, permissions,
        system_prompt=build_system_prompt(
            root=ws.root, state=state, persona_prompt=persona.prompt if persona else "",
            memory_block=memory_prompt_block(memory, str(ws.root)), skills_block=skills.catalog_text(allowed_skills)),
        cancel=cancel, approver=approver, on_event=on_event,
        on_message=session.append, on_rewrite=session.rewrite,
        audit=AuditLog(state / "audit.jsonl", session.id),
        messages=messages,
        compaction_state=CompactionState.from_dict(session.meta.get("compaction")),
        usage=session.meta.get("usage") or None,
        config=EngineConfig(max_iterations=settings.max_iterations, context_window=settings.context_window,
                            settings=settings.request_settings()),
        ask_user=ask_user, plan_approver=plan_approver,
        limit_result=ResultSpiller(state / "spill", session.id, ws),
    )
    holder["engine"] = engine
    engine.session_id = session.id
    engine.persona, engine.missing_tools = persona, missing
    engine.memory, engine.skills = memory, skills  # for /memory and /skills
    engine.on_checkpoint = lambda: session.save_meta(**engine.snapshot())
    return engine
