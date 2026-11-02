"""Command line: `minicw` (the interactive terminal), `minicw run --prompt ...`, `minicw doctor`.

    minicw                       new session in the current folder
    minicw --continue            reopen the newest session of this folder
    minicw --resume <id>         reopen a given session
    minicw run --prompt "..."    one task, no questions asked, answer on stdout (exit 0 = completed)
    minicw doctor                check keys, model, packages

Options that exist on every command: --backend --model --mode --workspace --max-iterations
--context-window --temperature --max-tokens --reasoning-effort --debug.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console

from .app import build_engine, make_provider
from .core.config import Settings, load_env, load_settings, state_dir
from .core.permissions import Mode, allow_all, deny_all
from .core.personas import get_persona
from .core.provider import BACKENDS, ProviderError
from .core.sessions import Session, is_safe_session_id
from .ui.render import Renderer


def common_options() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--backend", choices=sorted(BACKENDS), default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--mode", choices=[m.value for m in Mode], default=None)
    p.add_argument("--workspace", "-w", default=".", help="folder the agent works in (default: here)")
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument("--context-window", type=int, default=None, help="the model's real window in tokens")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--persona", default=None, help="role to play: coder, assistant, or one of yours in <state>/personas")
    p.add_argument("--debug", action="store_true", help="log requests and replies (redacted) to <state>/debug.jsonl")
    return p


def build_parser(command: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minicw" + ("" if command == "chat" else f" {command}"),
                                     parents=[common_options()])
    if command in ("chat", "run"):
        parser.add_argument("--resume", metavar="ID", default=None, help="reopen this session")
        parser.add_argument("--continue", dest="continue_", action="store_true",
                            help="reopen the newest session of this workspace")
    if command == "run":
        parser.add_argument("--prompt", required=True)
    return parser


def apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    for name in ("backend", "model", "mode", "max_iterations", "context_window", "temperature",
                 "max_tokens", "reasoning_effort"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(settings, name, value)
    return settings


def open_or_create_session(args: argparse.Namespace, settings: Settings, state: Path,
                           workspace: Path) -> tuple[Session, list[dict]]:
    """The session to use, and its messages. The workspace is stored as the resolved path."""
    ws = str(workspace)
    if getattr(args, "resume", None):
        if not is_safe_session_id(args.resume):
            raise SystemExit(f"error: {args.resume!r} is not a valid session id")
        try:
            return Session.open(state, args.resume)
        except KeyError:
            raise SystemExit(f"error: no session {args.resume}") from None
    if getattr(args, "continue_", False):
        found = Session.list(state, ws)
        if found:
            return Session.open(state, found[0]["id"])
    return Session.create(state, workspace=ws, backend=settings.backend, model=settings.model,
                          mode=settings.mode), []


def main(argv: Optional[list[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # "→ ✓ ✗ ●" must not crash a piped Windows console
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv.pop(0) if argv and argv[0] in ("run", "doctor", "chat") else "chat"
    args = build_parser(command).parse_args(argv)

    load_env()
    state = state_dir()
    settings = apply_overrides(load_settings(state), args)
    workspace = Path(os.path.realpath(args.workspace))
    if not workspace.is_dir():
        print(f"error: workspace {workspace} is not a folder", file=sys.stderr)
        return 2

    if command == "doctor":
        from .doctor import run_doctor
        return run_doctor(settings.resolved(), state, workspace)

    session, messages = open_or_create_session(args, settings, state, workspace)
    # A resumed session brings its own backend/model/mode unless the command line says otherwise.
    if args.backend is None and session.meta.get("backend"):
        settings.backend = session.meta["backend"]
    if args.model is None and session.meta.get("model"):
        settings.model = session.meta["model"]
    if args.mode is None and session.meta.get("mode"):
        settings.mode = session.meta["mode"]
    settings.resolved()
    try:
        provider = make_provider(settings.backend, state, debug=args.debug)
    except ProviderError as exc:
        print(f"error: {exc}\nrun `minicw doctor` to check the setup", file=sys.stderr)
        return 2
    persona = None
    persona_id = args.persona if args.persona is not None else session.meta.get("persona", "")
    if persona_id and persona_id != "none":
        try:
            persona = get_persona(state, persona_id)
        except KeyError as exc:
            print(f"error: {exc.args[0]}", file=sys.stderr)
            return 2
    session.save_meta(backend=settings.backend, model=settings.model, mode=settings.mode,
                      persona=persona.id if persona else "")

    if command == "run":
        return run_once(args.prompt, settings, state, workspace, provider, session, messages, persona)

    from .integrations import load_extra_tools
    from .ui.repl import Repl

    console = Console()
    extra_tools, mcp = load_extra_tools(state, console)
    return Repl(console=console, settings=settings, state=state, workspace=workspace, provider=provider,
                session=session, messages=messages, debug=args.debug, persona=persona,
                extra_tools=extra_tools, mcp=mcp).run()


def run_once(prompt: str, settings: Settings, state: Path, workspace: Path, provider, session: Session,
             messages: list[dict], persona=None) -> int:
    """Headless: progress on stderr, the answer on stdout. Nobody can answer a question, so "ask"
    means no (unless the mode is bypass, which allows everything but the hard floors)."""
    from .integrations import load_extra_tools

    console = Console(stderr=True)
    renderer = Renderer(console, show_reasoning=False)
    extra_tools, _mcp = load_extra_tools(state, console)
    approver = allow_all if Mode(settings.mode) == Mode.BYPASS else deny_all
    engine = build_engine(settings=settings, workspace=workspace, state=state, provider=provider,
                          session=session, messages=messages, on_event=renderer.on_event, approver=approver,
                          persona=persona, extra_tools=extra_tools)
    result = engine.run(prompt)
    if result.text:
        print(result.text)
    return 0 if result.status == "completed" else 3
