"""Allow / ask / deny for every tool call.

The decision comes from the tool's risk class and the current mode:

    mode    READ   EGRESS  WRITE_LOCAL            EXEC                      EXTERNAL
    plan    allow  allow   deny                   read-only commands only   deny
    ask     allow  ask     ask                    read-only cmds allow      ask
    auto    allow  allow   allow (in workspace)   ask                       ask
    bypass  allow  allow   allow                  allow                     allow

"ask" is turned into "allow" when the user already answered "always" for that tool (or for
that exact shell command) in this session. The MODE is checked first and grants second, so a
grant given in ask mode cannot let a write through in plan mode.

Floors that no mode lowers (as in the reference, `permissions.py` @ 59151e3, MIT): destructive
shell commands are denied; writes outside the workspace are denied; files that run later
(git hooks, CI, `.minicw/`) always ask, except in bypass mode.

Adapted from mcw `permissions.py`. New: the plan mode, the risk-class table, grants that can be
exported and restored with the session, and a `source` on each decision for the audit log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from ..tools.registry import ToolRegistry, ToolSpec
from .readonly import is_readonly_command, read_targets
from .risk import Risk
from .workspace import Workspace


class Verdict(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class Mode(str, Enum):
    PLAN = "plan"  # explore only; ends with propose_plan
    ASK = "ask"  # writes and unknown commands go to the user
    AUTO = "auto"  # workspace writes pass; commands and outside actions still ask
    BYPASS = "bypass"  # everything passes except the hard floors


class Answer(str, Enum):
    ONCE = "once"  # this call only
    ALWAYS = "always"  # this tool (for shell: this exact command) for the rest of the session
    DENY = "deny"


@dataclass
class Decision:
    verdict: Verdict
    reason: str = ""
    # Who decided, for the audit log: "mode", "risk", "rule", "grant" or "floor".
    source: str = ""


# The "approver" is a callback (tool name, arguments, decision) -> Answer, so the same engine
# runs with a person at the terminal and with nobody (headless: ask means no).
Approver = Callable[[str, dict, Decision], Answer]


def deny_all(_tool: str, _args: dict, _decision: Decision) -> Answer:
    """Headless default: nobody is there to approve, so ask means no."""
    return Answer.DENY


def allow_all(_tool: str, _args: dict, _decision: Decision) -> Answer:
    return Answer.ONCE


# Hard deny: never run these, even in bypass mode. Regexes applied to the lower-cased command:
# recursive delete (rm -rf, Remove-Item -Recurse, rd /s), disk format, shutdown, dd, fork bomb,
# download-and-run (curl ... | sh), git push --force, git reset --hard, git clean -f.
# A list of known patterns, not a guarantee against every dangerous command.
# From mcw `permissions.py` (itself from the reference's list), unchanged.
DESTRUCTIVE = [
    r"(?:^|[;&|(]\s*)(rm|ri|del|erase|rmdir|rd|remove-item)\b.*\s-(r|rf|fr|rec|recurse|recursive)\b",
    r"(?:^|[;&|(]\s*)rm\b.*\s-\w*r\w*\b", r"(?:^|[;&|(]\s*)(rd|rmdir)\b.*\s/s\b",
    r"remove-item\b.*\s-rec",
    r"\bformat\s+[a-z]:", r"\bmkfs\b", r"\bshutdown\b", r"\bdd\s+if=", r":\(\)\s*\{",
    r"(curl|wget|iwr|invoke-webrequest)\b.*\|\s*(sh|bash|iex|invoke-expression)",
    r"\bgit\s+push\b.*(--force|\s-f\b|\s\+\S)", r"\bgit\s+reset\s+--hard\b", r"\bgit\s+clean\b.*\s-\w*f",
]

# Files inside the workspace that something runs later (a git hook, a CI job, settings). They
# can be edited, but never without a person saying yes (except in bypass mode). Case-insensitive
# because NTFS is. List as in the reference, `permissions.py:150-166` @ 59151e3 (MIT);
# `.coworker/` became `.minicw/`.
PROTECTED_IN_PROJECT = (
    ".git/hooks/",
    ".git/config",
    ".github/workflows/",
    ".gitlab-ci.yml",
    ".vscode/tasks.json",
    ".minicw/",
)


def is_protected_in_project(rel_path: str) -> bool:
    """`rel_path`: path relative to the workspace root, with "/" separators."""
    posix = "/" + rel_path.replace("\\", "/").lower().lstrip("/")
    return any(
        f"/{marker}" in posix if marker.endswith("/") else posix.endswith("/" + marker)
        for marker in PROTECTED_IN_PROJECT
    )


# A read target like "env:" or "HKCU:" is a PowerShell drive (or a URL), not a workspace file:
# `type env:DEEPSEEK_API_KEY` would print a key.
_PS_DRIVE = re.compile(r"^[A-Za-z][A-Za-z0-9_]+:")

PLAN_DENY = "plan mode: nothing may be changed. Explore, then call propose_plan with your plan."


class PermissionEngine:
    def __init__(self, registry: ToolRegistry, workspace: Workspace, mode: Mode = Mode.ASK) -> None:
        self.registry = registry
        self.workspace = workspace
        self.mode = mode
        # What the user answered "always" to. Lives for the session and is saved with it.
        self._session_tools: set[str] = set()
        self._session_commands: set[str] = set()

    # -- grants ---------------------------------------------------------------------------

    def remember(self, tool: str, arguments: dict[str, Any]) -> None:
        """Apply an "always" answer. For shell: only this exact command, not the whole tool,
        or one harmless "always" would open the door to every later command."""
        if tool == "run_shell":
            self._session_commands.add(str(arguments.get("command", "")))
        else:
            self._session_tools.add(tool)

    def grants(self) -> dict[str, list[str]]:
        return {"tools": sorted(self._session_tools), "commands": sorted(self._session_commands)}

    def load_grants(self, data: dict[str, Any] | None) -> None:
        data = data or {}
        self._session_tools = {str(t) for t in data.get("tools", [])}
        self._session_commands = {str(c) for c in data.get("commands", [])}

    # -- decision -------------------------------------------------------------------------

    def evaluate(self, tool: str, arguments: dict[str, Any]) -> Decision:
        spec = self.registry.get(tool)
        if spec is None:
            return Decision(Verdict.DENY, f"unknown tool: {tool}", "rule")
        if tool == "run_shell":
            return self._evaluate_shell(str(arguments.get("command", "")))
        if spec.writes:
            # A write needs a path, and it must stay in the workspace: in EVERY mode, bypass
            # included. (str(None) would be "None" and pass as a file name.)
            path = arguments.get("path")
            if not isinstance(path, str) or not path:
                return Decision(Verdict.DENY, "a write without a path", "floor")
            try:
                resolved = self.workspace.resolve(path, write=True)
            except (ValueError, OSError) as exc:
                return Decision(Verdict.DENY, str(exc), "floor")
            if self.mode == Mode.PLAN:
                return Decision(Verdict.DENY, PLAN_DENY, "mode")
            if self.mode != Mode.BYPASS and is_protected_in_project(self.workspace.rel(resolved)):
                return Decision(Verdict.ASK, "this file runs automatically later", "floor")
        return self._by_mode(spec)

    def _ask_or_granted(self, tool: str, reason: str) -> Decision:
        if tool in self._session_tools:
            return Decision(Verdict.ALLOW, "allowed for this session", "grant")
        return Decision(Verdict.ASK, reason, "risk")

    def _by_mode(self, spec: ToolSpec) -> Decision:
        risk, mode = spec.risk, self.mode
        if mode == Mode.BYPASS:
            return Decision(Verdict.ALLOW, "bypass mode", "mode")
        if risk == Risk.READ:
            return Decision(Verdict.ALLOW, "read-only", "risk")
        if mode == Mode.PLAN:
            if risk == Risk.EGRESS:
                return Decision(Verdict.ALLOW, "reads the web", "mode")
            return Decision(Verdict.DENY, PLAN_DENY, "mode")
        label = f"{risk.value} tool"
        if risk == Risk.EGRESS:
            return Decision(Verdict.ALLOW, "reads the web", "mode") if mode == Mode.AUTO \
                else self._ask_or_granted(spec.name, label)
        if risk == Risk.WRITE_LOCAL and mode == Mode.AUTO and not spec.always_ask:
            return Decision(Verdict.ALLOW, "write in workspace", "mode")
        return self._ask_or_granted(spec.name, label)

    # -- shell ----------------------------------------------------------------------------

    def _evaluate_shell(self, command: str) -> Decision:
        # 1. Destructive patterns are denied in every mode, bypass included.
        lowered = command.lower()
        for pattern in DESTRUCTIVE:
            if re.search(pattern, lowered):
                return Decision(Verdict.DENY, "destructive command", "floor")
        # 2. A single read-only command whose targets stay in the workspace: always fine.
        if is_readonly_command(command) and self._reads_stay_in_workspace(command):
            return Decision(Verdict.ALLOW, "read-only command", "rule")
        # 3. Plan mode stops here: only read-only commands are allowed.
        if self.mode == Mode.PLAN:
            return Decision(Verdict.DENY, PLAN_DENY, "mode")
        if self.mode == Mode.BYPASS:
            return Decision(Verdict.ALLOW, "bypass mode", "mode")
        # 4. The user said "always" for exactly this command.
        if command in self._session_commands:
            return Decision(Verdict.ALLOW, "command allowed for this session", "grant")
        return Decision(Verdict.ASK, "runs a shell command", "risk")

    def _reads_stay_in_workspace(self, command: str) -> bool:
        """True when every file the command would read is inside what the file tools may read.
        expanduser turns "~" into the home folder; without it "~/x" would be taken as a
        sub-folder called "~". PowerShell drives (env:, HKCU:) are never plain files."""
        from pathlib import Path

        for target in read_targets(command):
            if _PS_DRIVE.match(target):
                return False
            try:
                self.workspace.resolve(str(Path(target).expanduser()))
            except (ValueError, OSError, RuntimeError):  # RuntimeError: `~nosuchuser`
                return False
        return True
