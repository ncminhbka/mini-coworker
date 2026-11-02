"""Tools that talk to the person: ask_user and propose_plan.

They are registered so the model sees their schemas, but the ENGINE answers them (on the main
thread, where the terminal prompt lives); the functions below only run when no callback is
wired. Both follow the reference's `tools/ask.py` and `tools/plan.py` @ 59151e3 (MIT), reduced
to the single-question form.
"""

from __future__ import annotations

from ..core.risk import Risk
from .registry import ToolSpec, obj


def _unavailable(**_kw) -> dict:
    return {"error": "this tool is not available here"}


def make_interact_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            "ask_user",
            "Ask the user ONE question when you need something only they know (a choice, a missing "
            "detail) and cannot find in the files. Give `options` when the answer is a choice. Do not "
            "ask what you can find out yourself, and do not ask for permission to do your task.",
            obj({
                "question": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"},
                            "description": "Optional list of choices; the user may also type another answer."},
            }, ["question"]),
            _unavailable, Risk.READ,
        ),
        ToolSpec(
            "propose_plan",
            "Plan mode only: present your plan to the user for approval, when you understand the task. "
            "Write it as a short numbered list. If the user approves, you can carry it out; if not, you "
            "receive their feedback.",
            obj({"plan": {"type": "string", "description": "The plan, as a short numbered list."}}, ["plan"]),
            _unavailable, Risk.READ,
        ),
    ]
