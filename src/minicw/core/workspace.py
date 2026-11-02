"""The folder the agent works in, and the only place it may write.

Adapted from mcw `tools.py` (`Workspace`), which was in turn the reference's rule that every
path a file tool touches must resolve inside the session's folder. New here: extra READ-ONLY
roots, so a tool result that was spilled to the state folder can still be opened by `read_file`.
"""

from __future__ import annotations

from pathlib import Path


class Workspace:
    def __init__(self, root: str | Path, read_roots: list[str | Path] | None = None) -> None:
        # resolve() makes the path absolute and follows symlinks and "..".
        self.root = Path(root).resolve()
        self.read_roots = [Path(r).resolve() for r in (read_roots or [])]

    def add_read_root(self, path: str | Path) -> None:
        p = Path(path).resolve()
        if p not in self.read_roots:
            self.read_roots.append(p)

    @staticmethod
    def _inside(full: Path, base: Path) -> bool:
        return full == base or base in full.parents

    def resolve(self, path: str, *, write: bool = False) -> Path:
        """Absolute path of `path`, or ValueError when it is outside what the tool may touch.
        Relative paths are taken from the workspace root. The join is resolved BEFORE the check
        so that ".." is processed first. Writes are limited to the workspace; reads may also
        use the read-only roots."""
        p = Path(path)
        full = (p if p.is_absolute() else self.root / p).resolve()
        if self._inside(full, self.root):
            return full
        if not write and any(self._inside(full, r) for r in self.read_roots):
            return full
        where = "the workspace" if write else "the workspace or a read-only folder"
        raise ValueError(f"path is outside {where}: {path}")

    def rel(self, path: Path) -> str:
        """Path relative to the workspace root with "/" separators on every OS, so what the
        model sees is short and the same on Windows and Linux. Outside paths stay absolute."""
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return str(path)
