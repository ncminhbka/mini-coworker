"""Risk classes: what kind of effect a tool call can have.

Every tool must declare one (there is deliberately no default: a tool nobody classified
would otherwise pass as harmless). The permission engine decides from the risk class and the
mode, see `permissions.py`. Idea from OpenWorker `risk.py:18-28` @ 59151e3 (MIT).
"""

from __future__ import annotations

from enum import Enum


class Risk(str, Enum):
    READ = "read"  # reads local state only (files, git, clock)
    EGRESS = "egress"  # reaches out to the network to read (web search, web fetch)
    WRITE_LOCAL = "write_local"  # changes local files or local state that outlives the turn
    EXEC = "exec"  # runs arbitrary programs (shell)
    EXTERNAL = "external"  # acts on the outside world (send mail, edit calendar, MCP servers)
