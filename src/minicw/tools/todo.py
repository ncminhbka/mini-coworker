"""todo_write: the model's own task list for the current work.

From mcw `tools.py` (`make_todo_tool`), copied nearly verbatim from the reference
(`tools/todo.py:56-77` @ 59151e3, MIT). The argument is called `todos`, never `items`: a
top-level key named "items" shadows `.items()` in the chat templates of some providers and
makes every later request that replays the call fail with HTTP 400 (reference `todo.py:16-20`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.risk import Risk
from .registry import ToolSpec, obj

TODO_STATUSES = {"pending", "in_progress", "done"}


@dataclass
class TodoList:
    items: list[dict] = field(default_factory=list)


def make_todo_tool(todo: TodoList) -> ToolSpec:
    def todo_write(todos: list = None, items: list = None) -> dict:
        # Replaces the WHOLE list (the model always sends the full list). `items` is accepted as
        # an alias because some models send it, but the schema only advertises `todos`.
        normalized = []
        for entry in (todos if todos is not None else items) or []:
            if isinstance(entry, dict):
                status = entry.get("status", "pending")
                if status == "completed":  # a common alias for our "done"
                    status = "done"
                normalized.append({"content": str(entry.get("content", "")),
                                   "status": status if status in TODO_STATUSES else "pending"})
            else:  # a bare string is the content of a pending item
                normalized.append({"content": str(entry), "status": "pending"})
        todo.items = normalized
        return {"count": len(normalized), "todos": normalized}

    # READ: it only changes the in-memory list, not files, so it also works in plan mode.
    return ToolSpec(
        "todo_write",
        "Replace the task list. Provide the full list of todos each call.",
        obj({"todos": {"type": "array", "items": obj({
            "content": {"type": "string"},
            "status": {"type": "string", "enum": ["pending", "in_progress", "done"]},
        }, ["content", "status"])}}, ["todos"]),
        todo_write, Risk.READ,
    )
