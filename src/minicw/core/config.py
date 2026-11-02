"""Where state lives, and the settings that can be changed without touching code.

State folder: `$MINICW_STATE_DIR`, else `~/.minicw`. It holds sessions, memories, the audit log,
personas, skills, MCP config and Google tokens. Keys are read from the environment, and from
`.env` files: first the one in the current folder, then the one in the state folder (a value
that is already set always wins, so a real environment variable beats both files).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional

# The names this project has been using (see mcw/pricing.py and docs/local-model.md): DeepSeek's flash model,
# the local Qwen3.5-4B built from ollama/Modelfile.qwen35-4b (num_ctx 16384), and gpt-oss-20b on Groq.
# Change them with --model, /model, or `model = "..."` in <state>/config.toml.
DEFAULT_MODELS = {"deepseek": "deepseek-flash", "ollama": "mcw-qwen3.5-4b", "groq": "openai/gpt-oss-20b"}
# A model's real context window, in tokens. Compaction triggers at 80% of it. A small local
# model whose server silently cuts the prompt needs this set honestly (--context-window).
DEFAULT_CONTEXT = {"deepseek": 128_000, "ollama": 16_384, "groq": 131_072}


def state_dir() -> Path:
    path = Path(os.environ.get("MINICW_STATE_DIR") or Path.home() / ".minicw")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_dotenv(path: Path) -> None:
    """Put KEY=VALUE lines from a .env file into os.environ without overriding what is set.
    Values are never printed or logged."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_env() -> None:
    load_dotenv(Path.cwd() / ".env")
    # MINICW_STATE_DIR may itself come from the first file, so the state folder is found after it.
    load_dotenv(state_dir() / ".env")


@dataclass
class Settings:
    backend: str = "deepseek"
    model: str = ""
    mode: str = "ask"
    max_iterations: int = 100
    context_window: int = 0  # 0 = the backend's default
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None

    def request_settings(self) -> dict[str, Any]:
        """The sampling options that were actually set, to be passed to the provider."""
        pairs = (("temperature", self.temperature), ("max_tokens", self.max_tokens),
                 ("reasoning_effort", self.reasoning_effort))
        return {k: v for k, v in pairs if v is not None}

    def resolved(self) -> "Settings":
        """Fill the blanks that depend on the backend."""
        if not self.model:
            self.model = DEFAULT_MODELS.get(self.backend, "")
        if not self.context_window:
            self.context_window = DEFAULT_CONTEXT.get(self.backend, 32_000)
        return self


def load_settings(directory: Optional[Path] = None) -> Settings:
    """Settings from `<state>/config.toml` if it exists. Unknown keys are ignored."""
    path = (directory or state_dir()) / "config.toml"
    data: dict[str, Any] = {}
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        pass
    known = {f.name for f in fields(Settings)}
    return Settings(**{k: v for k, v in data.items() if k in known})
