"""explore(task): a helper agent that reads a lot and reports back briefly.

Why: a sub-task that needs many reads (ten long files, several web pages) would flood the main
conversation; the helper reads them in its OWN context and only its short report comes back.
Shallow on purpose, as in the reference (`tools/subagent.py:79-138` @ 59151e3, MIT) and mcw
(`subagent.py`): the helper gets only read-only tools, cannot delegate again, stops after 10
steps, shares the provider and model, and does not see the main conversation.

Differences that come from this being a terminal app with Ctrl-C:
* it shares the parent's `cancel` event, so one Ctrl-C stops both;
* it runs inside the parent's tool thread, so it cannot ask the user anything: its approver
  says no. Web access in a helper therefore needs auto mode or an "always" grant for the web
  tools given in the main conversation;
* only its tool calls are shown (marked as the helper's); its text is not.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ..core.engine import Engine, EngineConfig
from ..core.permissions import PermissionEngine, deny_all
from ..core.prompt import build_system_prompt
from ..core.risk import Risk
from ..core.workspace import Workspace
from .registry import ToolRegistry, ToolSpec, obj

if TYPE_CHECKING:
    pass

CHILD_MAX_ITERATIONS = 10
# Read-only by risk, but they change state the parent owns, or make no sense in a helper.
NOT_FOR_CHILD = {"explore", "ask_user", "propose_plan", "todo_write", "save_skill"}
CHILD_PROMPT = (
    "You are a helper working for another agent. Do only the sub-task below, using your tools. "
    "Do not ask questions. Finish with a short report: what you found, with the exact sources "
    "(file paths and line numbers, or URLs), and anything you could not find."
)


def make_explore_tool(get_parent: Callable[[], Engine], ws: Workspace, state: Path) -> ToolSpec:
    def explore(task: str) -> dict:
        parent = get_parent()
        registry = ToolRegistry()
        for spec in parent.registry.specs():
            if spec.read_only and spec.name not in NOT_FOR_CHILD:
                registry.register(spec)
        permissions = PermissionEngine(registry, ws, parent.permissions.mode)
        permissions.load_grants(parent.permissions.grants())

        def forward(event: dict) -> None:
            if event.get("type") in ("tool_proposed", "tool_finished"):
                parent.on_event({**event, "subagent": True})

        child = Engine(
            parent.provider, parent.model, registry, permissions,
            system_prompt=build_system_prompt(root=ws.root, state=state, base=CHILD_PROMPT),
            cancel=parent.cancel, approver=deny_all, on_event=forward,
            config=EngineConfig(max_iterations=CHILD_MAX_ITERATIONS, context_window=parent.config.context_window,
                                settings=dict(parent.config.settings)),
            limit_result=parent.limit_result,
        )
        result = child.run(task)
        for key in ("input", "output", "cache_read", "turns"):  # the helper's tokens count too
            parent.usage[key] += child.usage[key]
        out = {"status": result.status, "report": result.text, "iterations": result.iterations}
        if result.status != "completed":
            out["note"] = f"the helper stopped early ({result.status}); the report may be partial"
        if result.error:
            out["error"] = result.error
        elif not (result.text or "").strip():
            out["error"] = "the helper finished without a report; do the sub-task yourself or rephrase it"
        return out

    return ToolSpec(
        "explore",
        "Hand a self-contained sub-task (e.g. 'read these files and report where X is configured') to a "
        "helper with read-only tools. Only its short report comes back, so your own context stays small. "
        "The helper cannot see your conversation: put everything it needs in `task`.",
        obj({"task": {"type": "string"}}, ["task"]),
        explore, Risk.READ,
    )
