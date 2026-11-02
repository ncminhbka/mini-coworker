"""Keeping huge tool results out of the conversation.

A tool result stays in the history and is paid for again on every later turn. When one is
longer than the limit, the full text is written to a file in the state folder and the model gets
the beginning, the end, and the path, so it can read the middle on demand with `read_file`.

Idea and shape from the reference (`toolresult.py`, OPE-186 @ 59151e3, MIT). The spill folder
is added to the workspace as a READ-ONLY root, so `read_file` can open it but nothing can write
there. `read_file` results are never spilled (they are already a window the model chose; a
spilled spill would loop): they are cut in the middle instead.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..core.models import ToolCall
from ..core.redact import redact
from ..core.workspace import Workspace

DEFAULT_LIMIT = 10_000


def clip(text: str, limit: int) -> str:
    """Head + marker + tail (the end of a log or traceback is usually the useful part)."""
    if limit <= 0 or len(text) <= limit:
        return text
    head, tail = int(limit * 0.6), int(limit * 0.4)
    return f"{text[:head]}\n[... {len(text) - head - tail} characters omitted ...]\n{text[-tail:]}"


class ResultSpiller:
    def __init__(self, directory: Path, session_id: str, workspace: Workspace, limit: int = DEFAULT_LIMIT) -> None:
        self.directory = directory / session_id
        self.limit = limit
        workspace.add_read_root(directory)

    def __call__(self, call: ToolCall, text: str) -> str:
        if len(text) <= self.limit:
            return text
        if call.name == "read_file":
            return clip(text, self.limit) + "\n(read fewer lines at a time with start_line / max_lines)"
        # The call id comes from the model: make it a harmless file name.
        name = re.sub(r"[^A-Za-z0-9_-]", "_", call.id)[:64] or "call"
        path = self.directory / f"{name}.txt"
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            path.write_text(redact(text), encoding="utf-8", newline="")
        except OSError:
            return clip(text, self.limit)  # no disk, no spill: still bounded
        head, tail = text[: int(self.limit * 0.5)], text[-int(self.limit * 0.3):]
        omitted = len(text) - len(head) - len(tail)
        return (f"{head}\n[... {omitted} characters omitted. The full result ({len(text)} characters) is saved at "
                f"{path} : open it with read_file(path=..., start_line=..., max_lines=...) ...]\n{tail}")
