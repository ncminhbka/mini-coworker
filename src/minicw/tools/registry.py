"""Tool description and registry.

From mcw `tools.py` (ToolSpec, ToolRegistry, `_obj`), itself following the reference's
`tools/registry.py`. Differences: `risk` is a required `Risk`, not a free string with a "low"
default, and the registry refuses a spec that does not have one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..core.risk import Risk


@dataclass
class ToolSpec:
    name: str
    # Text the model reads to decide when to use the tool. It is part of the prompt.
    description: str
    # JSON Schema of the arguments (see `obj`).
    parameters: dict[str, Any]
    # The function that really runs, called as func(**arguments).
    func: Callable[..., Any]
    # What kind of effect the tool can have. No default, on purpose.
    risk: Risk
    # True when the tool writes to the path in its `path` argument: the permission engine then
    # checks that path against the workspace (outside = denied in every mode).
    writes: bool = False
    # True when even auto mode must ask (e.g. save_skill); "always" grants still apply.
    always_ask: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.risk, Risk):
            raise TypeError(f"tool {self.name!r}: risk must be a Risk, got {self.risk!r}")

    @property
    def read_only(self) -> bool:
        """Safe to hand to a read-only helper agent (explore)."""
        return self.risk in (Risk.READ, Risk.EGRESS)

    def schema(self) -> dict[str, Any]:
        """The element of the request's `tools` field (Chat Completions format)."""
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


def obj(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    """JSON Schema of an arguments object. Schemas are written by hand, not derived from
    function signatures, so the text the model sees is under our control."""
    return {"type": "object", "properties": properties, "required": required}


class ToolRegistry:
    def __init__(self) -> None:
        # dict keeps insertion order, so the tool order in the prompt is stable between runs.
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        self._tools[spec.name] = spec  # same name replaces the old tool
        return spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def keep_only(self, names: set[str]) -> None:
        self._tools = {n: t for n, t in self._tools.items() if n in names}

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def specs(self) -> list[ToolSpec]:
        return list(self._tools.values())
