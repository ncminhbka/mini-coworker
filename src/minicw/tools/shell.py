"""run_shell: run one command in the workspace, interruptibly.

From mcw `tools.py` (`make_shell_tool`), which copied the reference's environment filter,
non-interactive defaults and process-tree kill (`tools/shell.py`,
`sandbox/runner/executor.py` @ 59151e3, MIT). Changes for a REPL where Ctrl-C must work:

* the command waits in short slices (`communicate(timeout=0.25)`) and looks at the cancel
  event between them;
* on cancel or timeout the whole process TREE is killed while the shell is still alive (once
  the parent has died, `taskkill /T` can no longer find its children);
* on Windows the child starts in its own process group, so the console's Ctrl-C reaches only
  our handler, not the command.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time

from ..core.risk import Risk
from ..core.workspace import Workspace
from .registry import ToolSpec, obj

SHELL_TIMEOUT = 120  # default seconds
MAX_TIMEOUT = 600
SLICE = 0.25  # how often a waiting command checks for cancel

# Environment variables whose name looks like a secret are not passed to commands: the app loads
# API keys from .env into os.environ, and without this `echo $env:DEEPSEEK_API_KEY` would put the
# key in the transcript.
_SECRET_ENV = re.compile(r"(API_?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.I)
# Defaults so a command does not hang asking a question (reference `executor.py:48-53`).
_NONINTERACTIVE_ENV = {"GIT_TERMINAL_PROMPT": "0", "DEBIAN_FRONTEND": "noninteractive",
                       "PYTHONUNBUFFERED": "1", "PIP_NO_INPUT": "1"}


def shell_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not _SECRET_ENV.search(k)}
    return {**env, **_NONINTERACTIVE_ENV}


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill the command and every process it started. Killing only the shell would leave a
    child (a dev server, say) holding the output pipes, and the tool would hang."""
    if proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        proc.kill()


def make_shell_tool(ws: Workspace, cancel: threading.Event) -> ToolSpec:
    def run_shell(command: str, description: str = "", timeout_seconds: int = SHELL_TIMEOUT) -> dict:
        try:  # `description` only rides along for the user
            timeout = min(max(int(timeout_seconds or SHELL_TIMEOUT), 1), MAX_TIMEOUT)
        except (TypeError, ValueError):
            return {"error": f"timeout_seconds must be a whole number of seconds, got {timeout_seconds!r}"}
        # `powershell -Command` reports 1 for any failed last command; the trailer restores the
        # native exit code (and 1 for a failed cmdlet).
        win_trailer = "\n$ok = $?; if ($null -ne $LASTEXITCODE) { exit $LASTEXITCODE }; if (-not $ok) { exit 1 }"
        # Each call is a fresh process, so `cd` does not carry over to the next command.
        argv = (
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command + win_trailer]
            if sys.platform == "win32"
            else ["/bin/bash", "-c", command]
        )
        proc = subprocess.Popen(
            argv, cwd=ws.root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            stdin=subprocess.DEVNULL,  # a command that asks for input gets EOF at once
            encoding="utf-8", errors="replace", env=shell_env(),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
            start_new_session=sys.platform != "win32",
        )
        deadline = time.monotonic() + timeout
        stdout = stderr = ""
        why = ""
        while True:
            try:
                stdout, stderr = proc.communicate(timeout=SLICE)
                break
            except subprocess.TimeoutExpired:
                if cancel.is_set():
                    why = "interrupted by the user"
                elif time.monotonic() >= deadline:
                    why = f"timed out after {timeout}s"
                if why:
                    kill_tree(proc)
                    try:
                        stdout, stderr = proc.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass  # a grandchild still holds the pipes: give up on its output rather than hang
                    return {"error": f"{why}; the command and its children were killed", "exit_code": None,
                            "stdout": (stdout or "")[-2000:], "stderr": (stderr or "")[-2000:]}
        # Only the END of long output is kept: results and errors are usually at the end.
        out = {"exit_code": proc.returncode, "stdout": stdout[-8000:], "stderr": stderr[-4000:]}
        if len(stdout) > 8000 or len(stderr) > 4000:
            out["truncated"] = "only the last part of the output is shown"
        return out

    shell_name = "PowerShell" if sys.platform == "win32" else "bash"
    return ToolSpec(
        "run_shell",
        f"Run a {shell_name} command in the workspace and return exit code, stdout and stderr. "
        "Each call is a fresh process; `cd` does not persist. Never put a multi-line script "
        "inline: write it to a file, then run the file.",
        obj({
            "command": {"type": "string", "description": "The command line."},
            "description": {"type": "string", "description": "Why you are running it, in a few words."},
            "timeout_seconds": {"type": "integer", "description": f"Seconds (default {SHELL_TIMEOUT}, max {MAX_TIMEOUT})."},
        }, ["command"]),
        run_shell, Risk.EXEC,
    )
