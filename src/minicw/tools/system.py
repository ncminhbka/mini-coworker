"""Read-only tools about the machine and the repository: git_status, git_diff, git_log,
current_time, runtime_context.

`git_log` is local in the reference (`tools/git.py`); `git_status` / `git_diff` come from its
aisuite toolkit. `current_time` is a tool, not text in the prompt, so the system prompt stays
byte-identical and the prompt cache valid (reference `clock.py`). `runtime_context` tells the
model what it can rely on (git, python, node...) before it tries a command that cannot work
(reference `runtime_context.py`, which reads no secrets). All of them are Risk.READ.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from ..core.risk import Risk
from ..core.workspace import Workspace
from .registry import ToolSpec, obj

GIT_TIMEOUT = 15
MAX_GIT_CHARS = 20_000

PROGRAMS = ("git", "python", "pip", "node", "npm", "npx", "uv", "docker", "gh")
MARKER_FILES = ("README.md", "pyproject.toml", "package.json", "requirements.txt", "Cargo.toml", "go.mod",
                "Makefile", "AGENTS.md", ".gitignore", "Dockerfile")


def _git(ws: Workspace, args: list[str]) -> dict:
    try:
        done = subprocess.run(["git", *args], cwd=ws.root, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=GIT_TIMEOUT, stdin=subprocess.DEVNULL)
    except FileNotFoundError:
        return {"error": "git is not installed or not on PATH"}
    except subprocess.TimeoutExpired:
        return {"error": f"git took more than {GIT_TIMEOUT}s"}
    if done.returncode != 0:
        return {"error": (done.stderr or done.stdout).strip()[:500] or f"git exited with {done.returncode}"}
    out = done.stdout
    result = {"output": out[:MAX_GIT_CHARS]}
    if len(out) > MAX_GIT_CHARS:
        result["truncated"] = f"output cut at {MAX_GIT_CHARS} characters of {len(out)}"
    return result


def make_system_tools(ws: Workspace) -> list[ToolSpec]:
    def git_status() -> dict:
        return _git(ws, ["status", "--short", "--branch"])

    def git_diff(path: str = "", staged: bool = False) -> dict:
        args = ["diff", "--no-color"] + (["--staged"] if staged else [])
        if path:
            ws.resolve(path)  # must stay inside the workspace (ValueError is reported by the engine)
            args += ["--", path]
        return _git(ws, args)

    def git_log(max_count: int = 10, path: str = "") -> dict:
        args = ["log", f"-{min(max(int(max_count), 1), 50)}", "--date=short", "--pretty=format:%h %ad %an  %s"]
        if path:
            ws.resolve(path)
            args += ["--", path]
        return _git(ws, args)

    def current_time() -> dict:
        now = datetime.now().astimezone()
        return {"local": now.isoformat(timespec="seconds"), "weekday": now.strftime("%A"),
                "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "timezone": now.strftime("%z")}

    def runtime_context() -> dict:
        found = {name: shutil.which(name) for name in PROGRAMS}
        return {
            "workspace": str(ws.root),
            "platform": f"{platform.system()} {platform.release()}",
            "python": sys.version.split()[0],
            "shell_for_run_shell": "PowerShell" if sys.platform == "win32" else "bash",
            "programs_on_path": sorted(name for name, path in found.items() if path),
            "programs_missing": sorted(name for name, path in found.items() if not path),
            "project_files": [m for m in MARKER_FILES if (ws.root / m).exists()],
            "is_git_repo": (ws.root / ".git").exists(),
        }

    return [
        ToolSpec("git_status", "Show the git branch and the changed files (git status --short --branch). Read-only.",
                 obj({}, []), git_status, Risk.READ),
        ToolSpec("git_diff", "Show uncommitted changes (git diff), optionally for one path or the staged changes. Read-only.",
                 obj({"path": {"type": "string", "description": "Limit to this path."},
                      "staged": {"type": "boolean", "description": "Show staged changes instead."}}, []),
                 git_diff, Risk.READ),
        ToolSpec("git_log", "Show recent commits (hash, date, author, subject), optionally for one path. Read-only.",
                 obj({"max_count": {"type": "integer", "description": "How many (default 10, max 50)."},
                      "path": {"type": "string"}}, []),
                 git_log, Risk.READ),
        ToolSpec("current_time", "The current local date, time, weekday and timezone. Use it for anything that "
                                 "depends on 'today' or 'now'.", obj({}, []), current_time, Risk.READ),
        ToolSpec("runtime_context", "What this machine has: OS, shell, which programs (git, python, node...) are "
                                    "installed, and which project files exist. Call it before running a command "
                                    "that needs a program you are not sure about.",
                 obj({}, []), runtime_context, Risk.READ),
    ]
