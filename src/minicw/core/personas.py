"""Personas: a role for the assistant, as a folder with a `manifest.md`.

    ---
    id: coder
    name: Coder
    tools: [read_file, grep, run_shell]      # optional: narrows the tools (empty list = error)
    skills: [review-checklist]               # optional: which skills load_skill may open
    ---
    The body of the file is the persona's own prompt, added to the base instructions.

A persona picks the tools and skills a session gets and gives it a voice. The harness's own
tools (todo_write, ask_user, propose_plan, load_skill, save_skill, memory, explore) are always
kept. Built-in personas ship in the package; yours go in `<state>/personas/<id>/manifest.md` and
win over a built-in with the same id. A persona may have a `skills/` folder of its own.

Format and the rule that a broken manifest is an error (never a silent default that would give
the persona every tool) follow the reference (`personas/manifest.py:205-218,396-405` @ 59151e3,
MIT); only the fields above are read here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..tools.skills import as_list, split_frontmatter

_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# Tools that belong to the harness, not to a job: a persona never takes them away.
HARNESS_TOOLS = {"todo_write", "ask_user", "propose_plan", "load_skill", "save_skill",
                 "remember", "memory_read", "memory_update", "memory_forget", "memory_search", "explore"}


@dataclass
class Persona:
    id: str
    name: str
    prompt: str
    tools: list[str] = field(default_factory=list)  # empty = all registered tools
    skills: list[str] = field(default_factory=list)  # empty = all skills found
    path: Optional[Path] = None

    @property
    def skills_dir(self) -> Optional[Path]:
        return self.path / "skills" if self.path else None


def load_persona(directory: Path) -> Persona:
    manifest = directory / "manifest.md"
    text = manifest.read_text(encoding="utf-8").lstrip("﻿")
    meta, body = split_frontmatter(text)
    if not text.startswith("---") or not meta:
        raise ValueError(f"{manifest}: missing or unterminated '---' frontmatter")
    if "tools" in meta and not as_list(meta["tools"]):
        raise ValueError(f"{manifest}: 'tools:' is empty; list the tools, or drop the key for all tools")
    persona_id = meta.get("id") or directory.name
    if not _ID.match(persona_id):
        raise ValueError(f"{manifest}: id {persona_id!r} must be lowercase letters, digits, '-' or '_'")
    return Persona(id=persona_id, name=meta.get("name") or persona_id, prompt=body.strip(),
                   tools=as_list(meta.get("tools", "")), skills=as_list(meta.get("skills", "")), path=directory)


def builtin_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "personas"


def find_personas(state: Path) -> tuple[dict[str, Persona], dict[str, str]]:
    """(personas by id, problems by folder). A broken manifest does not hide the others."""
    found: dict[str, Persona] = {}
    problems: dict[str, str] = {}
    for root in (builtin_dir(), state / "personas"):  # the user's folder is read last, so it wins
        if not root.is_dir():
            continue
        for sub in sorted(root.iterdir()):
            if (sub / "manifest.md").is_file():
                try:
                    persona = load_persona(sub)
                    found[persona.id] = persona
                except (OSError, ValueError) as exc:
                    problems[str(sub)] = str(exc)
    return found, problems


def get_persona(state: Path, persona_id: str) -> Persona:
    found, problems = find_personas(state)
    if persona_id in found:
        return found[persona_id]
    detail = f"; problems: {'; '.join(problems.values())}" if problems else ""
    raise KeyError(f"no persona {persona_id!r} (have: {', '.join(sorted(found)) or 'none'}){detail}")
