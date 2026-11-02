"""The interactive terminal: a prompt, one engine turn per line, approvals, slash commands.

`prompt_toolkit` reads the line (history in a file, completion of /commands); `rich` prints.
Rules that keep the two from fighting each other:
* nothing is printed while a prompt is open (no background thread writes to the terminal);
* every question to the user during a turn (permission, ask_user, plan approval) is a
  `prompt_toolkit` prompt on the main thread, never `input()`: on Windows Ctrl-C then arrives
  as KeyboardInterrupt from the prompt, which the engine turns into "no, and stop the turn";
* Ctrl-C at the main prompt only clears the line; Ctrl-D or /exit leaves.

The turn itself is `Engine.run(line)`; the engine installs its own SIGINT handler for the
duration of the turn (see `core/engine.py`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ..app import build_engine, make_provider
from ..core.config import Settings
from ..core.engine import Engine
from ..core.permissions import Answer, Decision
from ..core.personas import Persona
from ..core.provider import OpenAICompatProvider
from ..core.sessions import Session
from .commands import COMMANDS, handle
from .render import Renderer, short


class SlashCompleter(Completer):
    """Completes /commands at the start of the line."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/") and " " not in text:
            for name in COMMANDS:
                if name.startswith(text):
                    yield Completion(name, start_position=-len(text), display_meta=COMMANDS[name][1])


class PlainPrompter:
    """Fallback when prompt_toolkit cannot use the terminal (Git Bash / mintty on Windows, input
    piped from a file): the same `prompt()` interface on top of input(). No history, no completion."""

    def prompt(self, message, **_kw) -> str:
        if callable(message):  # prompt_toolkit accepts a function that builds the prompt
            message = message()
        text = "".join(part for _style, part in message) if isinstance(message, FormattedText) else str(message)
        return input(text)


def make_prompters(history_file: str):
    """(main prompt, question prompt). prompt_toolkit when the terminal allows it, else plain."""
    try:
        return (PromptSession(history=FileHistory(history_file), completer=SlashCompleter()), PromptSession())
    except Exception:  # noqa: BLE001 - NoConsoleScreenBufferError and friends: any failure means "use input()"
        plain = PlainPrompter()
        return plain, plain


class Repl:
    def __init__(self, *, console: Console, settings: Settings, state: Path, workspace: Path,
                 provider: OpenAICompatProvider, session: Session, messages: list[dict],
                 debug: bool = False, persona: Optional[Persona] = None, extra_tools: Optional[list] = None,
                 mcp=None) -> None:
        self.console = console
        self.settings = settings
        self.state = state
        self.workspace = workspace
        self.provider = provider
        self.debug = debug
        self.persona = persona
        self.extra_tools: list = list(extra_tools or [])  # MCP and Google tools
        self.mcp = mcp  # the McpManager, for /mcp
        self.renderer = Renderer(console)
        # `asker` is for questions during a turn (permission, ask_user, plan): no history.
        self.prompter, self.asker = make_prompters(str(state / "history"))
        self.session = session
        self.engine: Engine = self._engine(session, messages)

    # -- engine lifecycle ------------------------------------------------------------------

    def _engine(self, session: Session, messages: list[dict]) -> Engine:
        return build_engine(
            settings=self.settings, workspace=self.workspace, state=self.state, provider=self.provider,
            session=session, messages=messages, on_event=self.renderer.on_event, approver=self.approve,
            ask_user=self.ask_user, plan_approver=self.approve_plan,
            persona=self.persona, extra_tools=self.extra_tools,
        )

    def rebuild(self) -> None:
        """A new engine on the SAME session and history, to pick up a changed persona."""
        self.engine.on_checkpoint()
        self.engine = self._engine(self.session, self.engine.messages)
        self.banner()

    def reload_integrations(self) -> None:
        """Restart the MCP servers, re-check Google, and rebuild the engine with the new tools."""
        from ..integrations import load_extra_tools

        self.extra_tools, self.mcp = load_extra_tools(self.state, self.console, self.mcp)
        self.rebuild()

    def switch(self, session: Session, messages: list[dict]) -> None:
        """A new engine on another session (/new, /resume). The system prompt is rendered once
        per engine, so a fresh one is the way to pick up a changed prompt."""
        self.engine.on_checkpoint()
        if session is not self.session:
            self.session.discard_if_empty()  # leaving a session nobody wrote to: do not keep a husk
        self.session = session
        self.engine = self._engine(session, messages)
        self.banner()

    # -- questions during a turn (main thread) ----------------------------------------------

    def _ask(self, message: str) -> str:
        return self.asker.prompt(FormattedText([("bold cyan", message)])).strip()

    def approve(self, tool: str, arguments: dict, decision: Decision) -> Answer:
        self.renderer._close()
        what = Text(tool, style="bold") + Text("  " + short(arguments, 300), style="dim")
        self.console.print(Panel(what, title=f"permission needed · {decision.reason}", border_style="yellow"))
        scope = "this exact command" if tool == "run_shell" else f"{tool}"
        try:
            reply = self._ask(f"[y]es once / [a]lways ({scope}) / [N]o › ").lower()
        except (KeyboardInterrupt, EOFError):
            self.engine.cancel.set()  # Ctrl-C here means "no" and "stop the turn"
            return Answer.DENY
        return {"y": Answer.ONCE, "a": Answer.ALWAYS}.get(reply[:1], Answer.DENY)

    def ask_user(self, question: str, options: list[str]) -> Optional[str]:
        self.renderer._close()
        self.console.print(Panel(Text(question), title="the assistant asks", border_style="cyan"))
        for i, option in enumerate(options, 1):
            self.console.print(Text(f"  {i}. {option}"))
        hint = "number or your own answer, Enter to skip" if options else "your answer, Enter to skip"
        reply = self._ask(f"({hint}) › ")
        if not reply:
            return None
        if reply.isdigit() and 1 <= int(reply) <= len(options):
            return options[int(reply) - 1]
        return reply

    def approve_plan(self, plan: str) -> tuple[bool, str]:
        self.renderer._close()
        self.console.print(Panel(Text(plan), title="proposed plan", border_style="magenta"))
        reply = self._ask("approve the plan? [y]es / type feedback / Enter = no › ")
        if reply.lower() in ("y", "yes"):
            return True, ""
        return False, reply

    # -- display -----------------------------------------------------------------------------

    def banner(self) -> None:
        e = self.engine
        line = Text("minicw ", style="bold") + Text(
            f"· {self.settings.backend}/{e.model} · {e.permissions.mode.value} · {self.workspace}", style="dim")
        self.console.print(line)
        self.console.print(Text(f"session {self.session.id}"
                                + (f" · persona {self.persona.id}" if self.persona else "")
                                + (f" · {len(e.messages)} messages" if e.messages else "")
                                + "   /help for commands, Ctrl-D to leave", style="dim"))
        if e.missing_tools:
            self.console.print(Text("(this persona lists tools that are not available: "
                                    + ", ".join(e.missing_tools) + ")", style="dim yellow"))

    def prompt_text(self) -> FormattedText:
        return FormattedText([("bold", f"[{self.engine.model} · {self.engine.permissions.mode.value}] "),
                              ("bold ansicyan", "› ")])

    # -- main loop ---------------------------------------------------------------------------

    def run(self) -> int:
        self.banner()
        while True:
            try:
                line = self.prompter.prompt(self.prompt_text).strip()
            except KeyboardInterrupt:  # Ctrl-C at the prompt: just a fresh line
                continue
            except EOFError:  # Ctrl-D
                break
            if not line:
                continue
            if line.startswith("/"):
                if handle(self, line):
                    break
                continue
            self.engine.run(line)
            self.console.print()
        self.engine.on_checkpoint()
        self.session.discard_if_empty()
        return 0
