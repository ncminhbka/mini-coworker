"""Sessions: a conversation that survives the process and can be reopened.

Layout under the state folder (compare the reference `conversations.py`, which uses a SQLite
index plus one .jsonl per conversation; a small JSON file per session is enough here):

    sessions/<id>.jsonl   one message per line, written the moment the message exists
    sessions/<id>.json    metadata: workspace, model, mode, title, grants, compaction, usage

Rules that keep the file trustworthy:
* The transcript is append-only. Nothing in minicw ever makes the history shorter, so there is
  no "rewrite if shorter" path. The one rewrite (`rewrite`) is atomic and only used to repair
  a torn last line or an orphan in the middle.
* Metadata is written AFTER the messages it describes, so a crash can leave the metadata a
  little old but never pointing past the end of the transcript.
* The id becomes a file name, so it must be one plain path component.
* Everything written goes through `redact`.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .redact import redact, redact_json

_SAFE_ID = re.compile(r"\A[A-Za-z0-9_-]{1,128}\Z")
TITLE_CHARS = 60


def is_safe_session_id(session_id: Any) -> bool:
    """No separators, no dots: an id like "../../x" must not reach the file system."""
    return isinstance(session_id, str) and bool(_SAFE_ID.match(session_id))


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _atomic_write(path: Path, text: str) -> None:
    """Write to a temp file in the same folder, then replace: a crash leaves the old file or the
    new one, never half of each."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def title_from(message: dict[str, Any]) -> str:
    content = message.get("content")
    text = content if isinstance(content, str) else ""
    first = text.strip().splitlines()[0] if text.strip() else ""
    return first[:TITLE_CHARS]


class Session:
    def __init__(self, directory: Path, meta: dict[str, Any]) -> None:
        self.directory = directory
        self.meta = meta
        self.id: str = meta["id"]
        self.path_jsonl = directory / f"{self.id}.jsonl"
        self.path_meta = directory / f"{self.id}.json"

    # -- create / open / list --------------------------------------------------------------

    @classmethod
    def create(cls, state_dir: Path, *, workspace: str, backend: str, model: str, mode: str,
               persona: str = "", session_id: Optional[str] = None) -> "Session":
        directory = state_dir / "sessions"
        directory.mkdir(parents=True, exist_ok=True)
        session_id = session_id or new_session_id()
        if not is_safe_session_id(session_id):
            raise ValueError(f"unsafe session id: {session_id!r}")
        meta = {"id": session_id, "workspace": workspace, "backend": backend, "model": model,
                "mode": mode, "persona": persona, "title": "", "created_at": _now(),
                "updated_at": _now(), "grants": {}, "compaction": {}, "usage": {}}
        session = cls(directory, meta)
        session.save_meta()
        return session

    @classmethod
    def open(cls, state_dir: Path, session_id: str) -> tuple["Session", list[dict[str, Any]]]:
        """The session and its messages. KeyError when it does not exist, ValueError for a
        bad id. A torn last line (the process died mid-write) is dropped and the file repaired."""
        if not is_safe_session_id(session_id):
            raise ValueError(f"unsafe session id: {session_id!r}")
        directory = state_dir / "sessions"
        path_meta = directory / f"{session_id}.json"
        if not path_meta.is_file():
            raise KeyError(session_id)
        meta = json.loads(path_meta.read_text(encoding="utf-8"))
        meta["id"] = session_id  # the file name wins over whatever the file says
        session = cls(directory, meta)
        messages, clean = session._read_messages()
        if not clean:
            session.rewrite(messages)
        return session, messages

    @staticmethod
    def list(state_dir: Path, workspace: Optional[str] = None) -> list[dict[str, Any]]:
        """Metadata of every session (of one workspace if given), newest first."""
        directory = state_dir / "sessions"
        found = []
        for path in directory.glob("*.json") if directory.is_dir() else []:
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(meta, dict) and is_safe_session_id(meta.get("id")) \
                    and (workspace is None or meta.get("workspace") == workspace):
                found.append(meta)
        return sorted(found, key=lambda m: m.get("updated_at", ""), reverse=True)

    # -- transcript ------------------------------------------------------------------------

    def _read_messages(self) -> tuple[list[dict[str, Any]], bool]:
        """(messages, clean). Not clean when a line could not be read or the last line has no
        newline; the caller then rewrites the file."""
        if not self.path_jsonl.is_file():
            return [], True
        raw = self.path_jsonl.read_bytes()
        clean = raw == b"" or raw.endswith(b"\n")
        messages = []
        for line in raw.decode("utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError:
                clean = False  # a torn line: skip it
                continue
            if isinstance(value, dict):
                messages.append(value)
        return messages, clean

    def append(self, message: dict[str, Any]) -> None:
        """Write one message at once. Opening in append mode each time keeps this simple and
        survives a crash between messages."""
        with open(self.path_jsonl, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(redact_json(message) + "\n")
        if not self.meta.get("title") and message.get("role") == "user":
            self.meta["title"] = redact(title_from(message))

    def discard_if_empty(self) -> None:
        """Remove a session that never got a message (opened and closed without typing)."""
        try:
            if not self.path_jsonl.exists() or self.path_jsonl.stat().st_size == 0:
                self.path_jsonl.unlink(missing_ok=True)
                self.path_meta.unlink(missing_ok=True)
        except OSError:
            pass

    def rewrite(self, messages: list[dict[str, Any]]) -> None:
        """Atomic full rewrite: only for repairs (torn line, orphan in the middle)."""
        _atomic_write(self.path_jsonl, "".join(redact_json(m) + "\n" for m in messages))

    # -- metadata --------------------------------------------------------------------------

    def save_meta(self, **updates: Any) -> None:
        self.meta.update(updates)
        self.meta["updated_at"] = _now()
        _atomic_write(self.path_meta, redact_json(self.meta, indent=1))
