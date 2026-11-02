"""Audit log: one JSON line per tool call, saying what was asked, what was decided and by whom.

`<state>/audit.jsonl`, append-only, redacted. It answers "who allowed this, and why" after the
fact. The reference keeps a richer trail (`audit.py`, `provenance.py` @ 59151e3, MIT); this one
keeps the part that answers that question.

Fields: ts, session, tool, risk, arguments (shortened), verdict (allow/ask/deny), source
(mode/risk/rule/grant/floor/user), reason, outcome (ok/error/denied/interrupted).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .redact import redact_json

MAX_ARG_CHARS = 500  # an argument longer than this (a whole file body, say) is cut in the log


def _shorten(arguments: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in (arguments or {}).items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        out[key] = text if len(text) <= MAX_ARG_CHARS else text[:MAX_ARG_CHARS] + f"… ({len(text)} chars)"
    return out


class AuditLog:
    def __init__(self, path: Path, session_id: str = "") -> None:
        self.path = path
        self.session_id = session_id

    def record(self, *, tool: str, arguments: dict[str, Any], risk: Optional[str], verdict: str,
               source: str, reason: str = "", outcome: str = "") -> None:
        entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "session": self.session_id,
                 "tool": tool, "risk": risk, "arguments": _shorten(arguments), "verdict": verdict,
                 "source": source, "reason": reason, "outcome": outcome}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(redact_json(entry) + "\n")
        except OSError:
            pass  # a full disk must not stop the agent; the transcript still has the call
