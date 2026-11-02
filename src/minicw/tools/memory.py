"""Long-term memory: facts that outlive a session, kept in SQLite, shown in the system prompt.

From mcw `memory.py`, itself from the reference (`memory/base.py`, `sqlite_store.py`,
`tools.py` @ 59151e3, MIT). Two scopes: `global` (about the user) and `workspace` (about this
project, keyed by the workspace path). The reference's third scope, `session`, was never read
back and is not here.

* The block in the system prompt lists every memory in full while it is short; past
  INDEX_THRESHOLD_CHARS it switches to an index (the newest few in full, a one-line summary for
  the rest) and the model fetches bodies with `memory_read`.
* Search uses SQLite FTS5 (each word is a prefix, case and accents folded) and falls back to
  LIKE when the SQLite build has no FTS5.
* The block is rendered ONCE per engine, so the system prompt stays stable.
* Tools that write ask the user even in auto mode (`always_ask`): a wrong memory costs more
  than a missing one.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from ..core.risk import Risk
from .registry import ToolSpec, obj


class Scope(str, Enum):
    GLOBAL = "global"  # about the user; visible in every project
    WORKSPACE = "workspace"  # about this project (the workspace path)


@dataclass
class MemoryItem:
    id: int
    scope: Scope
    content: str
    summary: Optional[str] = None  # one line, used in index mode
    workspace: Optional[str] = None
    created_at: Optional[str] = None


class MemoryStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)  # tools run in worker threads
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope TEXT NOT NULL, content TEXT NOT NULL, summary TEXT,
                workspace TEXT, created_at TEXT NOT NULL)"""
        )
        self._conn.commit()
        # SQLite's lower() and LIKE only fold ASCII, so "Lương" would not match "lương". This
        # function folds with Python's casefold, for the LIKE fallback.
        self._conn.create_function("ufold", 1, lambda s: (s or "").casefold(), deterministic=True)
        self._fts = self._setup_fts()

    def _setup_fts(self) -> bool:
        """A FTS5 table next to `memories` (external content: it stores only the index) and three
        triggers that keep it in step. The default tokenizer, unicode61, folds case and removes
        accents, so "luong" finds "Lương". False when this SQLite has no FTS5."""
        try:
            existed = self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories_fts'").fetchone()
            self._conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5("
                               "content, summary, content=memories, content_rowid=id)")
            self._conn.execute(
                "CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN "
                "INSERT INTO memories_fts(rowid, content, summary) VALUES (new.id, new.content, new.summary); END")
            self._conn.execute(
                "CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN "
                "INSERT INTO memories_fts(memories_fts, rowid, content, summary) "
                "VALUES ('delete', old.id, old.content, old.summary); END")
            self._conn.execute(
                "CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN "
                "INSERT INTO memories_fts(memories_fts, rowid, content, summary) "
                "VALUES ('delete', old.id, old.content, old.summary); "
                "INSERT INTO memories_fts(rowid, content, summary) VALUES (new.id, new.content, new.summary); END")
            if not existed and self._conn.execute("SELECT count(*) FROM memories").fetchone()[0]:
                self._conn.execute("INSERT INTO memories_fts(memories_fts) VALUES ('rebuild')")
            self._conn.commit()
            return True
        except sqlite3.OperationalError:
            return False

    @staticmethod
    def _item(row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(id=row["id"], scope=Scope(row["scope"]), content=row["content"], summary=row["summary"],
                          workspace=row["workspace"], created_at=row["created_at"])

    def add(self, content: str, *, scope: Scope = Scope.WORKSPACE, summary: Optional[str] = None,
            workspace: Optional[str] = None) -> MemoryItem:
        cur = self._conn.execute(
            "INSERT INTO memories (scope, content, summary, workspace, created_at) VALUES (?,?,?,?,?)",
            (scope.value, content, summary, workspace if scope == Scope.WORKSPACE else None,
             datetime.now(timezone.utc).isoformat()))
        self._conn.commit()
        return self.get(cur.lastrowid)  # type: ignore[arg-type,return-value]

    def get(self, item_id: int) -> Optional[MemoryItem]:
        row = self._conn.execute("SELECT * FROM memories WHERE id=?", (item_id,)).fetchone()
        return self._item(row) if row else None

    def visible(self, workspace: Optional[str]) -> list[MemoryItem]:
        """What a session in `workspace` sees: all global memories and this workspace's. Oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE scope='global' OR (scope='workspace' AND workspace IS ?) ORDER BY id",
            (workspace,)).fetchall()
        return [self._item(r) for r in rows]

    def update(self, item_id: int, content: str, *, summary: Optional[str] = None) -> Optional[MemoryItem]:
        # COALESCE(?, summary): without a new summary the old one stays.
        self._conn.execute("UPDATE memories SET content=?, summary=COALESCE(?, summary) WHERE id=?",
                           (content, summary, item_id))
        self._conn.commit()
        return self.get(item_id)

    def delete(self, item_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM memories WHERE id=?", (item_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def search(self, query: str, *, limit: int = 5, workspace: Optional[str] = None) -> list[MemoryItem]:
        """Every word must match, as a prefix ("tok"*), ranked by relevance. A FTS syntax error or
        a missing FTS5 falls back to LIKE (reference `sqlite_store.py:194-242`)."""
        tokens = re.findall(r"\w+", query or "")
        if not tokens:
            return []
        if self._fts:
            try:
                rows = self._conn.execute(
                    "SELECT m.* FROM memories m JOIN memories_fts f ON m.id = f.rowid"
                    " WHERE memories_fts MATCH ? AND (m.scope='global' OR m.workspace IS ?)"
                    " ORDER BY f.rank LIMIT ?",
                    (" ".join(f'"{t}"*' for t in tokens), workspace, limit)).fetchall()
                return [self._item(r) for r in rows]
            except sqlite3.OperationalError:
                pass
        return self._search_like(tokens, limit=limit, workspace=workspace)

    def _search_like(self, tokens: list[str], *, limit: int, workspace: Optional[str]) -> list[MemoryItem]:
        def pattern(word: str) -> str:  # % and _ are escaped so they are not wildcards
            return "%" + word.casefold().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"

        clause = " AND ".join("ufold(content || ' ' || coalesce(summary,'')) LIKE ? ESCAPE '\\'" for _ in tokens)
        rows = self._conn.execute(
            f"SELECT * FROM memories WHERE {clause} AND (scope='global' OR workspace IS ?) ORDER BY id DESC LIMIT ?",
            [pattern(t) for t in tokens] + [workspace, limit]).fetchall()
        return [self._item(r) for r in rows]


# -- the block in the system prompt ----------------------------------------------------------

# When to remember and when not to. Without this the model either never calls `remember` or saves
# trivia. Ideas and order from the reference `agent.py:76-99` (_MEMORY_GUIDANCE) @ 59151e3 (MIT).
MEMORY_GUIDANCE = """\
Memory:
- You have persistent memory across sessions. Use `remember` for durable facts: the user's \
corrections and stated preferences (include the why), and project context you could not \
rederive from the files. Scope by what the fact is about: about the user -> "global"; about \
the current work -> "workspace". Always pass a one-line summary (15 words max).
- Save conservatively: a wrong memory costs more than a missing one. Save only clearly \
durable facts ("from now on", "always"). A one-off phrasing: apply it now, do not save it. \
When the user explicitly asks you to remember something, always save it.
- Sensitive topics (health, finances, relationships, beliefs): never save silently. Ask \
first and save only on a yes.
- When you save, say so in one short sentence in your visible reply.
- Do not save what the files already record or details that only matter to the current task. \
Use absolute dates, never "yesterday".
- Before saving, check the known memories: if one already covers it, revise it with \
`memory_update` instead of adding a near-duplicate; retire wrong or obsolete ones with \
`memory_forget`.
- Memories reflect when they were written. If one names a file or URL, verify it still \
exists before relying on it."""

INDEX_THRESHOLD_CHARS = 8_000
INDEX_FULL_NEWEST = 10
_INDEX_NOTE = ("(Some memories above show only a one-line summary. Call memory_read with the [#id]s "
               "before acting on anything a summary hints at.)")


def _one_line(item: MemoryItem) -> str:
    text = (item.summary or "").strip()
    if not text:
        first = item.content.strip().splitlines()
        text = first[0] if first else ""
        if len(text) > 80:
            text = text[:77] + "..."
    return f"- [#{item.id}] {text}"


def render_memory_block(items: list[MemoryItem], *, threshold_chars: int = INDEX_THRESHOLD_CHARS) -> str:
    """Full text while short, an index when long. Note: the index has no cap on its LENGTH (a
    thousand memories would still be a thousand lines); that is the reference's behaviour too."""
    if not items:
        return ""
    header = "Known memories (from earlier sessions):\n"
    full = header + "\n".join(f"- [#{i.id}] {i.content}" for i in items)
    if len(full) <= threshold_chars:
        return full
    newest = {i.id for i in sorted(items, key=lambda i: i.id)[-INDEX_FULL_NEWEST:]}
    lines = [f"- [#{i.id}] {i.content}" if i.id in newest else _one_line(i) for i in items]
    return header + "\n".join(lines) + "\n" + _INDEX_NOTE


def memory_prompt_block(store: MemoryStore, workspace: str) -> str:
    return MEMORY_GUIDANCE + "\n\n" + render_memory_block(store.visible(workspace))


# -- tools -----------------------------------------------------------------------------------


def make_memory_tools(store: MemoryStore, *, workspace: str) -> list[ToolSpec]:
    def remember(content: str, scope: str = "workspace", summary: str = "") -> dict:
        try:
            sc = Scope(scope)
        except ValueError:
            return {"error": f"scope must be one of {[s.value for s in Scope]}"}
        item = store.add(content, scope=sc, summary=summary or None, workspace=workspace)
        return {"id": item.id, "scope": item.scope.value}

    def memory_read(memory_ids: list) -> dict:
        # An id that does not exist (or is not a number) is reported, so the model knows
        # instead of believing it read everything.
        found, missing = [], []
        for mid in memory_ids:
            try:
                item = store.get(int(mid))
            except (TypeError, ValueError):
                item = None
            if item:
                found.append({"id": item.id, "scope": item.scope.value, "content": item.content})
            else:
                missing.append(mid)
        return {"memories": found, **({"missing": missing} if missing else {})}

    def memory_update(memory_id: int, content: str, summary: str = "") -> dict:
        item = store.update(int(memory_id), content, summary=summary or None)
        return {"id": item.id} if item else {"error": f"no memory #{memory_id}"}

    def memory_forget(memory_id: int) -> dict:
        return {"deleted": store.delete(int(memory_id))}

    def memory_search(query: str, limit: int = 5) -> dict:
        hits = store.search(query, limit=int(limit), workspace=workspace)
        return {"memories": [{"id": i.id, "content": i.content} for i in hits]}

    return [
        ToolSpec("remember",
                 "Save a durable fact or preference for future sessions. Scope: global (about the user) or "
                 "workspace (about this project). Prefer memory_update over a near-duplicate.",
                 obj({"content": {"type": "string"},
                      "scope": {"type": "string", "enum": [s.value for s in Scope]},
                      "summary": {"type": "string", "description": "One line, used in index mode."}}, ["content"]),
                 remember, Risk.WRITE_LOCAL, always_ask=True),
        ToolSpec("memory_read", "Fetch full memory bodies by id.",
                 obj({"memory_ids": {"type": "array", "items": {"type": "integer"}}}, ["memory_ids"]),
                 memory_read, Risk.READ),
        ToolSpec("memory_update", "Replace the text of an existing memory by id.",
                 obj({"memory_id": {"type": "integer"}, "content": {"type": "string"},
                      "summary": {"type": "string"}}, ["memory_id", "content"]),
                 memory_update, Risk.WRITE_LOCAL, always_ask=True),
        ToolSpec("memory_forget", "Delete a memory that is wrong or stale, by id.",
                 obj({"memory_id": {"type": "integer"}}, ["memory_id"]),
                 memory_forget, Risk.WRITE_LOCAL, always_ask=True),
        ToolSpec("memory_search", "Search saved memories by keywords.",
                 obj({"query": {"type": "string"}, "limit": {"type": "integer"}}, ["query"]),
                 memory_search, Risk.READ),
    ]
