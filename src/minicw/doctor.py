"""`minicw doctor`: check the setup before a demo.

Prints one line per check: ✓ fine, ✗ a problem that stops minicw from working, • something
optional that is missing. The model check sends one tiny request, so it costs almost nothing.
Values of keys are never printed, only whether they are set.
"""

from __future__ import annotations

import importlib
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.text import Text

from .core.config import Settings
from .core.provider import BACKENDS, OpenAICompatProvider, ProviderError

OK, BAD, OPT = "ok", "bad", "optional"
Check = tuple[str, str, str]  # (status, name, detail)


def _python() -> Check:
    ok = sys.version_info >= (3, 11)
    return (OK if ok else BAD, "python", f"{sys.version.split()[0]}" + ("" if ok else " (3.11 or newer is needed: tomllib)"))


def _state(state: Path) -> Check:
    try:
        probe = state / ".doctor"
        probe.write_text("x")
        probe.unlink()
        return OK, "state folder", f"{state} (writable)"
    except OSError as exc:
        return BAD, "state folder", f"{state}: {exc}"


def _key(settings: Settings) -> Check:
    import os

    env = BACKENDS[settings.backend][1]
    if env is None:
        return OK, f"{settings.backend} key", "none needed"
    return (OK, f"{env}", "is set") if os.environ.get(env) else (BAD, f"{env}", "is not set (put it in .env)")


def _ollama() -> Check:
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2):
            return OK, "ollama", "reachable on localhost:11434"
    except (urllib.error.URLError, OSError):
        return OPT, "ollama", "not running (only needed for /backend ollama)"


def _model(settings: Settings, state: Path) -> Check:
    try:
        provider = OpenAICompatProvider.for_backend(settings.backend)
    except ProviderError as exc:
        return BAD, f"model {settings.model}", str(exc)
    try:
        turn = provider.complete(model=settings.model, messages=[{"role": "user", "content": "Reply with the single word OK."}],
                                 max_tokens=8)
    except ProviderError as exc:
        return BAD, f"model {settings.backend}/{settings.model}", str(exc)[:200]
    return OK, f"model {settings.backend}/{settings.model}", f"answers ({(turn.text or '').strip()[:20]!r})"


def _package(name: str, optional: bool, hint: str) -> Check:
    try:
        importlib.import_module(name)
        return OK, f"package {name}", "installed"
    except ImportError:
        return (OPT if optional else BAD), f"package {name}", f"missing: {hint}"


def _program(name: str, hint: str) -> Check:
    path = shutil.which(name)
    return (OK, name, path) if path else (OPT, name, f"not found: {hint}")


def _mcp(settings: Settings, state: Path) -> Check:
    from .integrations.mcp import load_config

    servers, problems = load_config(state / "mcp.json")
    if problems:
        return BAD, "mcp.json", "; ".join(problems)
    if not servers:
        return OPT, "mcp servers", f"none configured ({state / 'mcp.json'})"
    return OK, "mcp servers", ", ".join(s.name for s in servers) + " (see /mcp for their state)"


def _google(settings: Settings, state: Path) -> Check:
    from .integrations.google.auth import status

    text = status(state)
    return (OK if text.startswith("signed in") else OPT), "google (gmail, calendar)", text


EXTRA_CHECKS: list[Callable[[Settings, Path], Check]] = [_mcp, _google]


def run_doctor(settings: Settings, state: Path, workspace: Path) -> int:
    console = Console()
    checks: list[Check] = [
        _python(), _state(state), _key(settings), _model(settings, state), _ollama(),
        _package("rich", False, "pip install rich"), _package("prompt_toolkit", False, "pip install prompt_toolkit"),
        _package("ddgs", False, "pip install ddgs (web_search)"),
        _package("googleapiclient", True, 'pip install "minicw[google]" (Gmail and Calendar)'),
        _program("git", "the git tools and the prompt's git snapshot need it"),
        _program("npx", "only for MCP servers written in Node"),
    ]
    checks += [check(settings, state) for check in EXTRA_CHECKS]
    failed = 0
    for status, name, detail in checks:
        mark, style = {OK: ("✓", "green"), BAD: ("✗", "bold red"), OPT: ("•", "yellow")}[status]
        failed += status == BAD
        console.print(Text(f" {mark} ", style=style) + Text(f"{name:<28}") + Text(detail, style="dim"))
    console.print()
    console.print(Text("all good" if not failed else f"{failed} problem(s) to fix", style="green" if not failed else "red"))
    return 1 if failed else 0
