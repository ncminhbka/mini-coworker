"""Shared dataclasses: what a provider returns and what the loop passes around."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenUsage:
    input: int = 0
    output: int = 0
    cache_read: int = 0

    @property
    def context_tokens(self) -> int:
        return self.input + self.cache_read

    def as_dict(self) -> dict[str, int]:
        return {"input": self.input, "output": self.output, "cache_read": self.cache_read}


@dataclass
class AssistantTurn:
    text: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: Optional[str] = None
    # Thinking text, and the response field it arrived in (`reasoning_content` for DeepSeek,
    # `reasoning` for Groq/OpenRouter). Replayed under the same field on later turns.
    reasoning: Optional[str] = None
    reasoning_field: Optional[str] = None
    usage: Optional[TokenUsage] = None


RAW_KEY = "_raw"  # arguments that never parsed as JSON are kept under this key
