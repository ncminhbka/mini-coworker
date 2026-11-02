"""Skills: written instructions for a kind of work, loaded when needed.

A skill is a folder with a SKILL.md: a small frontmatter (name, description) and a markdown
body. "Progressive disclosure": only the catalog (name + description) goes into the system
prompt, so many skills cost little; the body is read by the model with `load_skill` when the
work calls for it. `save_skill` lets the model propose a new skill, which the user approves.

From mcw `skills.py` (skill half), itself from the reference (`skills/base.py`, `store.py`
@ 59151e3, MIT). Where skills are searched, in order: `<state>/skills`, `<workspace>/.minicw/skills`,
and the active persona's own `skills/` folder; a later folder wins on a name clash.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..core.risk import Risk
from .registry import ToolSpec, obj

SKILL_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """(keys, body). The frontmatter is the block between two `---` lines at the top, one
    `key: value` per line (not full YAML). No frontmatter gives ({}, text)."""
    text = text.lstrip("﻿")  # editors on Windows may save a BOM
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    meta: dict[str, str] = {}
    for line in text[3:end].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip().lower()] = value.strip()
    return meta, text[end + 4:].lstrip("\n")


def as_list(value: str) -> list[str]:
    """"[a, b, c]" or "a, b, c" -> ["a", "b", "c"]."""
    return [v.strip() for v in value.strip().strip("[]").split(",") if v.strip()]


@dataclass
class Skill:
    name: str
    description: str
    instructions: str = ""
    path: Optional[str] = None
    allowed_tools: list[str] = field(default_factory=list)  # advice to the model, not enforced


class SkillLoader:
    def __init__(self, dirs: list[Path]) -> None:
        self.dirs = [Path(d) for d in dirs]
        self._skills: dict[str, Skill] = {}
        self.rescan()

    def rescan(self) -> None:
        self._skills = {}
        for directory in self.dirs:
            try:
                entries = sorted(directory.iterdir()) if directory.is_dir() else []
            except OSError:
                continue
            for sub in entries:
                md = sub / "SKILL.md"
                try:
                    if md.is_file():
                        skill = self._parse(md)
                        self._skills[skill.name] = skill
                except (OSError, UnicodeDecodeError):
                    continue  # one bad skill costs only itself

    @staticmethod
    def _parse(md: Path) -> Skill:
        meta, body = split_frontmatter(md.read_text(encoding="utf-8"))
        return Skill(name=meta.get("name") or md.parent.name, description=meta.get("description", ""),
                     instructions=body.strip(), path=str(md.parent),
                     allowed_tools=as_list(meta.get("allowed-tools", meta.get("allowed_tools", ""))))

    def names(self) -> list[str]:
        return list(self._skills)

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def catalog_text(self, allowed: Optional[set[str]] = None) -> str:
        lines = [f"- {s.name}: {s.description}" for s in self._skills.values() if allowed is None or s.name in allowed]
        if not lines:
            return ""
        return "Skills available (call load_skill with the name to read one before using it):\n" + "\n".join(lines)


def make_skill_tools(loader: SkillLoader, save_dir: Path, allowed: Optional[set[str]] = None) -> list[ToolSpec]:
    def load_skill(name: str) -> dict:
        skill = loader.get(name)
        if skill is None:
            loader.rescan()  # a skill saved after the session began must still load
            skill = loader.get(name)
        visible = [n for n in loader.names() if allowed is None or n in allowed]
        if skill is None or (allowed is not None and skill.name not in allowed):
            return {"error": f"no skill named {name!r}", "available": visible}
        out = {"name": skill.name, "instructions": skill.instructions, "path": skill.path}
        if skill.allowed_tools:
            out["suggested_tools"] = skill.allowed_tools
        return out

    def save_skill(name: str, description: str, instructions: str, overwrite: bool = False) -> dict:
        if not SKILL_NAME.match(name or ""):
            return {"error": "name must be lowercase letters, digits, '-' or '_' (max 63 characters)"}
        if not description.strip() or not instructions.strip():
            return {"error": "description and instructions must not be empty"}
        folder = save_dir / name
        target = folder / "SKILL.md"
        if target.exists() and not overwrite:
            return {"error": f"skill {name!r} already exists; pass overwrite=true to replace it"}
        folder.mkdir(parents=True, exist_ok=True)
        one_line = " ".join(description.split())
        target.write_text(f"---\nname: {name}\ndescription: {one_line}\n---\n{instructions.strip()}\n",
                          encoding="utf-8", newline="\n")
        loader.rescan()
        return {"saved": str(target), "note": "it will be in the catalog from the next session"}

    return [
        ToolSpec("load_skill", "Load the full instructions of a skill from the catalog by name.",
                 obj({"name": {"type": "string"}}, ["name"]), load_skill, Risk.READ),
        ToolSpec("save_skill",
                 "Propose saving a procedure that worked as a reusable skill: a short name, a one-line "
                 "description saying WHEN to use it, and the step-by-step instructions. The user must approve.",
                 obj({"name": {"type": "string"}, "description": {"type": "string"}, "instructions": {"type": "string"},
                      "overwrite": {"type": "boolean"}}, ["name", "description", "instructions"]),
                 save_skill, Risk.WRITE_LOCAL, always_ask=True),
    ]
