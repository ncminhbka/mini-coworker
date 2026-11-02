"""Slash commands of the REPL. `handle(repl, line)` returns True when the REPL should exit.

Each command is a small function (repl, argument) -> None. `COMMANDS` maps the name to
(function, help text); the help text is also what the completer shows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from rich.table import Table
from rich.text import Text

from ..app import make_provider
from ..core.config import DEFAULT_CONTEXT, DEFAULT_MODELS
from ..core.permissions import Mode
from ..core.personas import find_personas, get_persona
from ..core.provider import BACKENDS, ProviderError
from ..core.sessions import Session, is_safe_session_id

if TYPE_CHECKING:
    from .repl import Repl


def _say(repl: "Repl", text: str, style: str = "") -> None:
    repl.console.print(Text(text, style=style))


def cmd_help(repl: "Repl", arg: str) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    for name, (_fn, help_text) in COMMANDS.items():
        table.add_row(Text(name, style="bold"), Text(help_text))
    repl.console.print(table)


def cmd_new(repl: "Repl", arg: str) -> None:
    session = Session.create(repl.state, workspace=str(repl.workspace), backend=repl.settings.backend,
                             model=repl.engine.model, mode=repl.engine.permissions.mode.value)
    repl.switch(session, [])


def cmd_sessions(repl: "Repl", arg: str) -> None:
    found = Session.list(repl.state, str(repl.workspace))
    if not found:
        _say(repl, "no sessions for this workspace yet", "dim")
        return
    table = Table(box=None, padding=(0, 2))
    for column in ("", "id", "updated", "model", "title"):
        table.add_column(column, style="dim" if column in ("updated", "model") else "")
    for meta in found[:20]:
        mark = "*" if meta["id"] == repl.session.id else ""
        table.add_row(mark, meta["id"], meta.get("updated_at", "")[:16].replace("T", " "),
                      meta.get("model", ""), meta.get("title", "") or "(empty)")
    repl.console.print(table)
    _say(repl, "resume one with /resume <id>", "dim")


def cmd_resume(repl: "Repl", arg: str) -> None:
    if not arg:
        _say(repl, "usage: /resume <id>   (see /sessions)", "red")
        return
    if not is_safe_session_id(arg):
        _say(repl, "that is not a valid session id", "red")
        return
    try:
        session, messages = Session.open(repl.state, arg)
    except KeyError:
        _say(repl, f"no session {arg}", "red")
        return
    repl.switch(session, messages)


def cmd_mode(repl: "Repl", arg: str) -> None:
    names = [m.value for m in Mode]
    if arg not in names:
        _say(repl, f"usage: /mode {'|'.join(names)}   (now: {repl.engine.permissions.mode.value})", "red")
        return
    repl.engine.permissions.mode = Mode(arg)
    repl.engine.on_checkpoint()
    _say(repl, f"mode → {arg}", "cyan")


def cmd_model(repl: "Repl", arg: str) -> None:
    if not arg:
        _say(repl, f"model: {repl.engine.model}   (usage: /model <name>)")
        return
    repl.engine.model = arg
    repl.engine.on_checkpoint()
    _say(repl, f"model → {arg}", "cyan")


def cmd_backend(repl: "Repl", arg: str) -> None:
    """Switch the provider, e.g. /backend ollama [model]: the fallback when the network fails."""
    parts = arg.split()
    if not parts or parts[0] not in BACKENDS:
        _say(repl, f"usage: /backend {'|'.join(BACKENDS)} [model]", "red")
        return
    backend = parts[0]
    try:
        provider = make_provider(backend, repl.state, debug=repl.debug)
    except ProviderError as exc:
        _say(repl, f"cannot switch: {exc}", "red")
        return
    repl.provider = provider
    repl.settings.backend = backend
    repl.engine.provider = provider
    repl.engine.model = parts[1] if len(parts) > 1 else DEFAULT_MODELS[backend]
    repl.engine.config.context_window = DEFAULT_CONTEXT[backend]
    repl.session.save_meta(backend=backend)
    repl.engine.on_checkpoint()
    _say(repl, f"backend → {backend}, model → {repl.engine.model}", "cyan")


def cmd_tokens(repl: "Repl", arg: str) -> None:
    u = repl.engine.usage
    _say(repl, f"this session: {u['input']:,} input + {u['cache_read']:,} cached, {u['output']:,} output "
               f"tokens over {u['turns']} model calls")
    if repl.engine._last_context_tokens:
        _say(repl, f"last request: {repl.engine._last_context_tokens:,} tokens "
                   f"(window {repl.engine.config.context_window:,}, compaction at 80%)", "dim")


def cmd_persona(repl: "Repl", arg: str) -> None:
    """/persona lists; /persona <id> switches (same conversation, new system prompt); /persona none clears."""
    found, problems = find_personas(repl.state)
    if not arg:
        for pid, persona in sorted(found.items()):
            mark = "*" if repl.persona and repl.persona.id == pid else " "
            _say(repl, f" {mark} {pid:<12} {persona.name}  ({len(persona.tools) or 'all'} tools)")
        for folder, why in problems.items():
            _say(repl, f"   ! {folder}: {why}", "red")
        _say(repl, "switch with /persona <id>, clear with /persona none", "dim")
        return
    if arg == "none":
        repl.persona = None
    else:
        try:
            repl.persona = get_persona(repl.state, arg)
        except KeyError as exc:
            _say(repl, str(exc.args[0]), "red")
            return
    repl.session.save_meta(persona=repl.persona.id if repl.persona else "")
    repl.rebuild()


def cmd_memory(repl: "Repl", arg: str) -> None:
    """/memory lists what the assistant remembers here; /memory forget <id> deletes one."""
    memory = repl.engine.memory
    if arg.startswith("forget"):
        words = arg.split()
        if len(words) == 2 and words[1].isdigit():
            _say(repl, "forgotten" if memory.delete(int(words[1])) else "no such memory")
        else:
            _say(repl, "usage: /memory forget <id>", "red")
        return
    items = memory.visible(str(repl.workspace))
    if not items:
        _say(repl, "nothing remembered yet", "dim")
    for item in items:
        _say(repl, f"[#{item.id}] ({item.scope.value}) {item.summary or item.content[:100]}")


def cmd_skills(repl: "Repl", arg: str) -> None:
    repl.engine.skills.rescan()
    if not repl.engine.skills.names():
        _say(repl, f"no skills yet; put SKILL.md files under {repl.state / 'skills'} "
                   f"or {repl.workspace / '.minicw' / 'skills'}", "dim")
    for name in repl.engine.skills.names():
        _say(repl, f"{name:<20} {repl.engine.skills.get(name).description}")


def cmd_mcp(repl: "Repl", arg: str) -> None:
    """/mcp shows the servers from <state>/mcp.json; /mcp reload restarts them."""
    if arg == "reload":
        repl.reload_integrations()
        return
    mcp = repl.mcp
    if mcp is None or (not mcp.status and not mcp.problems):
        _say(repl, f"no MCP servers configured; list them in {repl.state / 'mcp.json'}", "dim")
        return
    for status in mcp.status.values():
        style = {"ready": "green", "failed": "red"}.get(status.state, "yellow")
        _say(repl, f"{status.name:<16} {status.state:<9} {status.detail or str(status.tools) + ' tools'}", style)
    for problem in mcp.problems:
        _say(repl, problem, "red")
    _say(repl, "tools are named mcp__<server>__<tool>; /mcp reload restarts the servers", "dim")


def cmd_google(repl: "Repl", arg: str) -> None:
    """/google shows the state of the Gmail/Calendar sign-in; /google login signs in (opens a browser)."""
    from ..integrations.google.auth import GoogleAuthError, login, status

    if arg == "login":
        _say(repl, "opening your browser to sign in to Google (waiting up to 3 minutes)…", "dim")
        try:
            login(repl.state)
        except GoogleAuthError as exc:
            _say(repl, str(exc), "red")
            return
        except Exception as exc:  # noqa: BLE001 - the browser step can fail in many ways
            _say(repl, f"sign-in failed: {type(exc).__name__}: {exc}", "red")
            return
        repl.reload_integrations()
        return
    _say(repl, status(repl.state))


def cmd_exit(repl: "Repl", arg: str) -> None:
    pass  # handled in handle()


COMMANDS: dict[str, tuple[Callable[["Repl", str], None], str]] = {
    "/help": (cmd_help, "show this list"),
    "/new": (cmd_new, "start a fresh session"),
    "/sessions": (cmd_sessions, "list the sessions of this workspace"),
    "/resume": (cmd_resume, "reopen a session: /resume <id>"),
    "/mode": (cmd_mode, "permission mode: plan | ask | auto | bypass"),
    "/model": (cmd_model, "change the model: /model <name>"),
    "/backend": (cmd_backend, "change the provider: /backend ollama|deepseek|groq [model]"),
    "/tokens": (cmd_tokens, "token use of this session"),
    "/persona": (cmd_persona, "list or switch personas: /persona [id|none]"),
    "/memory": (cmd_memory, "what is remembered here; /memory forget <id>"),
    "/skills": (cmd_skills, "skills the assistant can load"),
    "/mcp": (cmd_mcp, "MCP servers: status, or /mcp reload"),
    "/google": (cmd_google, "Gmail and Calendar sign-in: status, or /google login"),
    "/exit": (cmd_exit, "leave (also /quit or Ctrl-D)"),
}


def handle(repl: "Repl", line: str) -> bool:
    name, _, arg = line.partition(" ")
    if name in ("/exit", "/quit"):
        return True
    entry = COMMANDS.get(name)
    if entry is None:
        _say(repl, f"unknown command {name}; /help lists them", "red")
        return False
    entry[0](repl, arg.strip())
    return False
