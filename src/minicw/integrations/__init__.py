"""Tools that come from outside the program: MCP servers and Google."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.text import Text

from ..tools.registry import ToolSpec
from .google import make_google_tools
from .mcp import McpManager


def load_extra_tools(state: Path, console: Optional[Console] = None,
                     previous: Optional[McpManager] = None) -> tuple[list[ToolSpec], McpManager]:
    """Start the MCP servers from `<state>/mcp.json` (stopping the `previous` set first) and add
    the Google tools if they are set up. Prints one line per server when a console is given."""
    if previous is not None:
        previous.close()
    manager = McpManager(state)
    if (state / "mcp.json").is_file() and console is not None:
        console.print(Text("starting MCP servers…", style="dim"))
    manager.start()
    if console is not None:
        for status in manager.status.values():
            style = {"ready": "green", "failed": "red"}.get(status.state, "yellow")
            detail = f"{status.tools} tools" if status.state == "ready" else status.detail or status.state
            console.print(Text(f"  mcp {status.name}: ", style="dim") + Text(detail, style=style))
        for problem in manager.problems:
            console.print(Text(f"  mcp: {problem}", style="red"))
    return manager.tool_specs() + make_google_tools(state), manager
