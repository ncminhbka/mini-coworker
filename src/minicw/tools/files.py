"""File tools: read_file, list_files, grep, write_file, replace_in_file.

From mcw `tools.py` (`make_file_tools`), which copied them from the reference
(`tools/files.py`, `sandbox/runner/tools_read.py`, `tools_grep.py` @ 59151e3, MIT). Changes:
each tool has a `Risk`; writes use `Workspace.resolve(..., write=True)`.

Tools return a dict. A failure is a dict with an "error" key, not an exception, so the model
can read it and correct itself.
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any

from ..core.risk import Risk
from ..core.workspace import Workspace
from .registry import ToolSpec, obj

# Limits so one call cannot flood the model's context.
DEFAULT_MAX_LINES = 500  # read_file: lines per read
MAX_READ_LINES = 2000  # read_file: ceiling for max_lines
MAX_LINE_CHARS = 2000  # read_file: longer lines are cut
DEFAULT_MAX_RESULTS = 200  # list_files
GREP_MAX_FILE_BYTES = 2_000_000  # grep skips (and reports) bigger files
LINE_TRUNCATED_MARK = "… (line truncated)"
# Folders skipped when walking a tree: huge, or not source. The last ones are OS data
# (a workspace in the home folder would otherwise make grep walk into AppData).
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", "target", ".next",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".idea", "Library", "AppData", "Application Data",
}


def _walk_files(base: Path):
    """Every file under `base`, in name order (so results are the same on every run).
    Symlinks are not followed: a cloned repo can carry a link to ~/.ssh."""
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)  # in place, or os.walk ignores it
        for name in sorted(filenames):
            path = Path(dirpath, name)
            if not path.is_symlink():
                yield path


def _skipped(files: list[str]) -> dict[str, Any]:
    if not files:
        return {}
    return {"skipped_large": files[:20],
            "note": f"{len(files)} file(s) over {GREP_MAX_FILE_BYTES // 1_000_000} MB were not searched"}


def make_file_tools(ws: Workspace) -> list[ToolSpec]:
    def read_file(path: str, start_line: int = 1, max_lines: int = DEFAULT_MAX_LINES) -> dict:
        full = ws.resolve(path)
        if not full.is_file():
            return {"error": f"not a file: {path}"}
        # Models sometimes send numbers as strings, hence int().
        start = max(1, int(start_line))
        limit = min(max(int(max_lines), 1), MAX_READ_LINES)
        selected: list[str] = []
        total = 0
        # Streamed line by line: a log of several hundred MB does not hang the tool. Every line
        # is still counted so total_lines is right. errors="replace": bad bytes do not crash.
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                total = i
                if i < start or len(selected) >= limit:
                    continue
                text = line.rstrip("\n")
                if len(text) > MAX_LINE_CHARS:
                    text = text[:MAX_LINE_CHARS] + LINE_TRUNCATED_MARK
                selected.append(f"{i:>5}\t{text}")  # "   12<tab>text" so code can be cited path:line
        end = start + len(selected) - 1
        out: dict[str, Any] = {"path": ws.rel(full), "total_lines": total, "content": "\n".join(selected)}
        if end < total:
            out["note"] = f"showing lines {start}-{end} of {total}; pass start_line={end + 1} to continue"
        return out

    def list_files(path: str = ".", pattern: str = "*", recursive: bool = True,
                   max_results: int = DEFAULT_MAX_RESULTS) -> dict:
        base = ws.resolve(path)
        if not base.is_dir():
            return {"error": f"not a folder: {path}"}
        found: list[str] = []
        truncated = False
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            entries = [(d + "/", Path(dirpath, d)) for d in dirnames] + [
                (f, Path(dirpath, f)) for f in sorted(filenames)
            ]
            for name, full in entries:
                if fnmatch.fnmatch(name.rstrip("/"), pattern):
                    if len(found) >= int(max_results):
                        truncated = True
                        break
                    found.append(ws.rel(full) + ("/" if name.endswith("/") else ""))
            # os.walk goes top-down, so the first round is `base` itself.
            if truncated or not recursive:
                break
        return {"files": found, "truncated": truncated}

    def grep(pattern: str, path: str = ".", glob: str = "*", max_results: int = 100) -> dict:
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            return {"error": f"bad regex: {exc}"}
        base = ws.resolve(path)
        limit = min(int(max_results), 1000)
        hits: list[str] = []
        skipped_large: list[str] = []
        for file in [base] if base.is_file() else _walk_files(base):
            if not (fnmatch.fnmatch(file.name, glob) or fnmatch.fnmatch(ws.rel(file), glob)):
                continue
            try:
                if file.stat().st_size > GREP_MAX_FILE_BYTES:
                    skipped_large.append(ws.rel(file))
                    continue
                with open(file, "rb") as fh:
                    if b"\0" in fh.read(8192):  # binary
                        continue
                text = file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for n, line in enumerate(text.splitlines(), start=1):
                if rx.search(line):
                    hits.append(f"{ws.rel(file)}:{n}:{line[:300]}")
                    if len(hits) >= limit:
                        return {"matches": hits, "truncated": True, **_skipped(skipped_large)}
        return {"matches": hits, "truncated": False, **_skipped(skipped_large)}

    def write_file(path: str, content: str, overwrite: bool = True) -> dict:
        full = ws.resolve(path, write=True)
        if full.is_dir():
            return {"error": f"path is a directory: {path}"}
        if full.exists() and not overwrite:
            return {"error": f"file already exists: {path}"}
        full.parent.mkdir(parents=True, exist_ok=True)
        # newline="": write exactly what the model sent (Windows would turn "\n" into "\r\n").
        full.write_text(content, encoding="utf-8", newline="")
        return {"path": ws.rel(full), "bytes": len(content.encode("utf-8"))}

    def replace_in_file(path: str, old: str, new: str, expected_replacements: int = 1) -> dict:
        full = ws.resolve(path, write=True)
        if not full.is_file():
            return {"error": f"not a file: {path}"}
        if not old:
            return {"error": "old must be a non-empty string"}
        if int(expected_replacements) < 1:
            return {"error": "expected_replacements must be >= 1"}
        # newline="" on read and write keeps the file's own line endings (CRLF stays CRLF).
        try:
            with open(full, encoding="utf-8", newline="") as f:
                text = f.read()
        except UnicodeDecodeError:
            return {"error": f"{path} is not UTF-8 text; replace_in_file only edits text files"}
        if "\r\n" in text and "\r\n" not in old:
            old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
        count = text.count(old)
        # Replace only when the number of matches is what the model expected. Zero: the text
        # was mistyped. More than expected: the fragment is too generic. Either way, refuse.
        if count != int(expected_replacements):
            hint = "read the file again and copy it exactly" if count == 0 else (
                "include more context to make it unique, or set expected_replacements")
            return {"error": f"expected {expected_replacements} replacement(s), found {count}: {hint}"}
        updated = text.replace(old, new)
        full.write_text(updated, encoding="utf-8", newline="")
        return {"path": ws.rel(full), "replacements": count,
                "chars_before": len(text), "chars_after": len(updated)}

    path_prop = {"type": "string", "description": "Path relative to the workspace."}
    return [
        ToolSpec(
            "read_file",
            "Read a text file, returning numbered lines ('   12\\ttext') so code can be "
            "referenced as path:line. Large files are windowed: pass start_line to continue "
            "where the previous read stopped. Read-only.",
            obj({
                "path": path_prop,
                "start_line": {"type": "integer", "description": "First line, 1-based (default 1)."},
                "max_lines": {"type": "integer", "description": f"How many lines (default {DEFAULT_MAX_LINES})."},
            }, ["path"]),
            read_file, Risk.READ,
        ),
        ToolSpec(
            "list_files",
            "List files and folders under a path; folders end with '/'. Use recursive=false to "
            "see one level, and pattern (a glob such as '*.py') to filter by name. Read-only.",
            obj({
                "path": {"type": "string", "description": "Folder to list (default '.')."},
                "pattern": {"type": "string", "description": "Glob to match names (default '*')."},
                "recursive": {"type": "boolean", "description": "Descend into subfolders (default true)."},
                "max_results": {"type": "integer", "description": f"Stop after this many (default {DEFAULT_MAX_RESULTS})."},
            }, []),
            list_files, Risk.READ,
        ),
        ToolSpec(
            "grep",
            "Search the workspace for a regular-expression pattern and return matching lines "
            "as file:line:text. Read-only.",
            obj({
                "pattern": {"type": "string", "description": "Regular expression."},
                "path": {"type": "string", "description": "Subdirectory or file (default: whole workspace)."},
                "glob": {"type": "string", "description": "Optional filename glob, e.g. '*.py'."},
                "max_results": {"type": "integer", "description": "Max matches (default 100, max 1000)."},
            }, ["pattern"]),
            grep, Risk.READ,
        ),
        ToolSpec(
            "write_file",
            "Create a NEW file, or replace an existing file's ENTIRE content when most of it "
            "changes. For a small or medium change to an existing file, read it and use "
            "replace_in_file instead; the whole `content` stays in the conversation on every "
            "later turn, so rewriting large files is expensive.",
            obj({"path": path_prop, "content": {"type": "string", "description": "Full file content."},
                 "overwrite": {"type": "boolean", "description": "Replace an existing file (default true)."}},
                ["path", "content"]),
            write_file, Risk.WRITE_LOCAL, writes=True,
        ),
        ToolSpec(
            "replace_in_file",
            "Edit an existing file in place: replace `old` (an exact text fragment, whitespace "
            "included) with `new`. Prefer this over write_file for changing part of a file; read "
            "the file first so `old` matches exactly. Set expected_replacements when the "
            "fragment appears more than once.",
            obj({
                "path": path_prop,
                "old": {"type": "string", "description": "Exact text to replace."},
                "new": {"type": "string", "description": "Replacement text."},
                "expected_replacements": {"type": "integer", "description": "How many matches there must be (default 1)."},
            }, ["path", "old", "new"]),
            replace_in_file, Risk.WRITE_LOCAL, writes=True,
        ),
    ]
