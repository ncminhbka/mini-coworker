"""Turns the engine's events into what the terminal shows.

The engine reports progress as plain dicts through `on_event` (type + fields); this module is the
only place that knows how they look. Model text is printed with markup OFF (`Text`, or
`markup=False`): a reply containing "[/x]" would otherwise raise rich's MarkupError in the
middle of a stream. There is no `rich.Live` and no spinner: text is written as deltas arrive,
and nothing is printed while a prompt is open.
"""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.text import Text


def short(value: Any, limit: int = 100) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = text.replace("\n", "⏎")
    return text if len(text) <= limit else text[: limit - 1] + "…"


class Renderer:
    def __init__(self, console: Console, *, show_reasoning: bool = True) -> None:
        self.console = console
        self.show_reasoning = show_reasoning
        self._open = ""  # "", "text" or "reasoning": which kind of line is being streamed

    def _close(self) -> None:
        """End the line being streamed, if any."""
        if self._open:
            self.console.print()
            self._open = ""

    def _stream(self, kind: str, piece: str, style: str, header: Text) -> None:
        if self._open != kind:
            self._close()
            self.console.print(header, end="")
            self._open = kind
        self.console.print(piece, end="", style=style, markup=False, highlight=False, soft_wrap=True)

    def on_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "assistant_delta":
            self._stream("text", event["text"], "", Text("● ", style="bold green"))
        elif kind == "reasoning_delta":
            if self.show_reasoning:
                self._stream("reasoning", event["text"], "dim italic", Text("… ", style="dim"))
        elif kind == "assistant_message":
            self._close()
        elif kind == "tool_proposed":
            self._close()
            helper = "    ↳ " if event.get("subagent") else ""
            line = Text(helper + "→ ", style="yellow") + Text(event["name"], style="bold yellow")
            line.append(" " + short(event.get("arguments", {})), style="dim")
            self.console.print(line)
        elif kind == "tool_finished":
            ok = event.get("status") == "ok"
            style = "green" if ok else "red"
            helper = "    ↳" if event.get("subagent") else ""
            line = Text(helper + ("  ✓ " if ok else "  ✗ "), style=style) + Text(f"{event['name']} · {event.get('status')}", style=style)
            line.append("  " + short(event.get("preview", ""), 90), style="dim")
            self.console.print(line)
        elif kind == "compacted":
            self._close()
            self.console.print(Text("(older messages were summarised to save context)", style="dim"))
        elif kind == "mode_changed":
            self._close()
            self.console.print(Text(f"mode → {event['mode']}", style="cyan"))
        elif kind == "interrupted":
            self._close()
            self.console.print(Text("⏹ interrupted", style="bold red"))
        elif kind == "error":
            self._close()
            self.console.print(Text(f"error: {event.get('error')}", style="red"))
        elif kind == "turn_end":
            self._close()
            status = event.get("status")
            if status == "max_iterations_exceeded":
                self.console.print(Text("⚠ stopped: too many steps in one turn", style="red"))
            elif status == "truncated":
                self.console.print(Text("⚠ the reply kept getting cut off at the output limit", style="red"))
