"""One client for every backend: OpenAI Chat Completions over plain urllib, with streaming.

Ollama (`/v1`), DeepSeek and Groq all speak the same wire format, so one class covers them; only
base_url, key and model differ. `complete()` is one blocking round trip; `stream()` reads the
reply as server-sent events and calls `on_delta` while it arrives.

From mcw `provider.py` (itself following the reference's `providers/openai_provider.py` @
59151e3, MIT: the default output cap, dropping a parameter the server rejects, reasoning
replay). New here: `stream()`, which runs the HTTP read in a worker thread. Python cannot
interrupt a blocking socket read on Windows, so the calling thread polls a queue every 0.1 s and
can react to Ctrl-C (`cancel` event) while the worker is still stuck in the read.
"""

from __future__ import annotations

import http.client
import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

from .models import RAW_KEY, AssistantTurn, TokenUsage, ToolCall

# backend -> (base URL, name of the environment variable that holds the key, or None).
BACKENDS = {
    "ollama": ("http://localhost:11434/v1", None),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
}

# Output-token ceiling when nobody sets one. A ceiling, not a target. A server that refuses it
# gets one more request without it (see `_fix`).
DEFAULT_MAX_TOKENS = 32000
# Per-backend starting ceilings, when one is known. None are assumed: a server that refuses the default is
# asked again without it and the provider remembers (see `_send`), but a ceiling set too LOW would silently cut
# long replies, so nothing is guessed here.
BACKEND_MAX_TOKENS: dict[str, int] = {}
# Seconds to wait before retry 1, 2, 3 after a transient failure (Retry-After wins if longer).
RETRY_DELAYS: tuple[float, ...] = (1, 3, 8)
# Names under which providers return the model's thinking: DeepSeek `reasoning_content`,
# Groq / OpenRouter `reasoning`. The name is remembered so it is sent back under the same one.
REASONING_FIELDS = ("reasoning_content", "reasoning")

PERMANENT_ERROR_TEXT = ("api key", "authentication", "unauthorized", "insufficient", "quota",
                        "context length", "maximum context", "too long")
# A 400 that is really the endpoint rejecting the model's own malformed tool call (Groq
# answers `tool_use_failed`): the next sample is usually fine, so it is worth a retry.
MALFORMED_CALL_TEXT = ("tool_use_failed", "failed to call a function", "tool call validation",
                       "failed_generation", "output_parse_failed")


class ProviderError(Exception):
    """A provider or network failure. `status` is the HTTP code when there is one (429, 400,
    500...), `body` the server's text. The code tells a transient error from a permanent one."""

    def __init__(self, message: str, status: Optional[int] = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class StreamCancelled(Exception):
    """The stream was stopped by the user. `partial` holds what had arrived so far."""

    def __init__(self, partial: AssistantTurn) -> None:
        super().__init__("stream cancelled")
        self.partial = partial


# -- helpers shared by complete() and stream() ---------------------------------------------


def _retryable(exc: ProviderError) -> bool:
    text = f"{exc} {exc.body}".lower()
    if any(p in text for p in PERMANENT_ERROR_TEXT):
        return False
    if exc.status is None:  # connection failure, unreadable reply
        return True
    if exc.status in (408, 409, 429) or exc.status >= 500:
        return True
    return exc.status == 400 and any(p in text for p in MALFORMED_CALL_TEXT)


def _retry_after(value: Optional[str], limit: float = 30.0) -> Optional[float]:
    """Seconds the server asked us to wait (Retry-After header), capped. A date or a broken
    value is ignored and the normal back-off is used."""
    try:
        return min(max(float(value), 0.0), limit) if value else None
    except ValueError:
        return None


def wire_messages(messages: list[dict[str, Any]], *, replay_reasoning: bool = True,
                  keep_last: Optional[int] = None) -> list[dict[str, Any]]:
    """Only keys the Chat Completions schema knows are sent; our sidecars (usage, ts, status,
    finish_reason...) are dropped. An assistant message's thinking goes back under the field it
    arrived in, because reasoning models on compatible endpoints expect it on later turns.
    `keep_last` limits that to the last N assistant turns that have any (None = all)."""
    allowed = ("role", "content", "tool_calls", "tool_call_id", "name")
    thinking = [i for i, m in enumerate(messages)
                if m.get("role") == "assistant" and m.get("reasoning")
                and m.get("reasoning_field") in REASONING_FIELDS]
    replayed = set(thinking if keep_last is None else thinking[max(0, len(thinking) - keep_last):])
    out = []
    for i, m in enumerate(messages):
        wire = {k: m[k] for k in allowed if k in m}
        if replay_reasoning and i in replayed:
            wire[m["reasoning_field"]] = m["reasoning"]
        out.append(wire)
    return out


def parse_arguments(raw: Any) -> dict[str, Any]:
    """Tool-call arguments arrive as a JSON string. A string that is not a JSON object is kept
    under RAW_KEY instead of raising, so the engine can tell the model what went wrong."""
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {RAW_KEY: str(raw)}
    return parsed if isinstance(parsed, dict) else {RAW_KEY: str(raw)}


def parse_usage(u: Optional[dict[str, Any]]) -> Optional[TokenUsage]:
    """Token counts. Cached tokens are reported in one of two places depending on the provider
    (DeepSeek: `prompt_cache_hit_tokens`; OpenAI style: `prompt_tokens_details.cached_tokens`).
    `prompt_tokens` includes the cached part, so it is subtracted: `input` is then the part
    billed in full and `cache_read` the cheaper part."""
    if not u:
        return None
    cached = int(u.get("prompt_cache_hit_tokens")
                 or (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    return TokenUsage(input=max(int(u.get("prompt_tokens") or 0) - cached, 0),
                      output=int(u.get("completion_tokens") or 0), cache_read=cached)


def _make_calls(slots: list[dict[str, Any]]) -> list[ToolCall]:
    calls = []
    for i, s in enumerate(slots):
        # A provider that sends no id gets one by position plus a random part, so call_0 of this
        # turn never equals call_0 of an earlier turn (results are matched by id).
        calls.append(ToolCall(id=s.get("id") or f"call_{i}_{uuid.uuid4().hex[:8]}",
                              name=s.get("name", ""), arguments=parse_arguments(s.get("arguments"))))
    return calls


def _reasoning_of(message: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    for f in REASONING_FIELDS:
        if isinstance(message.get(f), str) and message[f]:
            return message[f], f
    return None, None


def parse_response(data: dict[str, Any]) -> AssistantTurn:
    """A non-streamed reply: {"choices": [{"message": {...}, "finish_reason": ...}], "usage": {...}}."""
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    slots = [{"id": tc.get("id"), "name": (tc.get("function") or {}).get("name", ""),
              "arguments": (tc.get("function") or {}).get("arguments")}
             for tc in (msg.get("tool_calls") or [])]
    reasoning, field = _reasoning_of(msg)
    return AssistantTurn(text=msg.get("content") or None, tool_calls=_make_calls(slots),
                         finish_reason=choice.get("finish_reason"), reasoning=reasoning,
                         reasoning_field=field, usage=parse_usage(data.get("usage")))


class StreamAccumulator:
    """Builds one AssistantTurn out of the chunks of a streamed reply.

    Text and thinking arrive as small deltas and are passed to `on_delta(kind, text)` at once.
    A tool call arrives in pieces (id and name first, then the JSON arguments in fragments);
    the pieces are joined per tool-call `index`. Some servers (Ollama) send every call with
    index 0 but a different id: a new non-empty id at an index that already has one starts a
    new call."""

    def __init__(self, on_delta: Optional[Callable[[str, str], None]] = None) -> None:
        self.on_delta = on_delta
        self.text = ""
        self.reasoning = ""
        self.field: Optional[str] = None
        self.finish_reason: Optional[str] = None
        self.usage_raw: Optional[dict[str, Any]] = None
        self.slots: list[dict[str, Any]] = []
        self._by_index: dict[int, dict[str, Any]] = {}

    def feed(self, chunk: dict[str, Any]) -> None:
        if chunk.get("error"):  # an error can arrive inside a 200 stream
            raise ProviderError(f"provider error in stream: {str(chunk['error'])[:300]}", None, json.dumps(chunk))
        if chunk.get("usage"):  # last chunk: has usage and "choices": []
            self.usage_raw = chunk["usage"]
        for choice in (chunk.get("choices") or [])[:1]:  # we ask for n=1
            delta = choice.get("delta") or {}
            for f in REASONING_FIELDS:
                piece = delta.get(f)
                if isinstance(piece, str) and piece:
                    self.reasoning += piece
                    self.field = self.field or f
                    if self.on_delta:
                        self.on_delta("reasoning", piece)
            piece = delta.get("content")
            if isinstance(piece, str) and piece:
                self.text += piece
                if self.on_delta:
                    self.on_delta("text", piece)
            for tc in delta.get("tool_calls") or []:
                self._tool_fragment(tc)
            if choice.get("finish_reason"):  # may come on an otherwise empty delta
                self.finish_reason = choice["finish_reason"]

    def _tool_fragment(self, tc: dict[str, Any]) -> None:
        index = tc.get("index", len(self.slots) - 1 if self.slots else 0)
        fn = tc.get("function") or {}
        tid = tc.get("id")
        slot = self._by_index.get(index)
        if slot is None or (tid and slot["id"] and tid != slot["id"]):
            slot = {"id": tid or "", "name": "", "arguments": ""}
            self.slots.append(slot)
            self._by_index[index] = slot
        elif tid and not slot["id"]:
            slot["id"] = tid
        if fn.get("name") and not slot["name"]:  # set once, never concatenated
            slot["name"] = fn["name"]
        args = fn.get("arguments")
        if isinstance(args, dict):
            args = json.dumps(args)
        if args:
            slot["arguments"] += args

    def partial(self) -> AssistantTurn:
        """What had arrived when the stream was stopped. Unfinished tool calls are dropped."""
        return AssistantTurn(text=self.text or None, finish_reason="interrupted",
                             reasoning=self.reasoning or None, reasoning_field=self.field)

    def finish(self) -> AssistantTurn:
        return AssistantTurn(text=self.text or None, tool_calls=_make_calls(self.slots),
                             finish_reason=self.finish_reason, reasoning=self.reasoning or None,
                             reasoning_field=self.field, usage=parse_usage(self.usage_raw))


def iter_sse(resp: Any, stop: threading.Event) -> Iterator[dict[str, Any]]:
    """Yield the JSON chunks of a server-sent-events body until `data: [DONE]`.
    Lines are split as bytes and decoded one by one (a multi-byte character can be cut by a
    network read, never by a line). Keep-alive comments (":...") and blank lines are skipped."""
    done = False
    for raw in iter(resp.readline, b""):
        if stop.is_set():
            return
        line = raw.strip()
        if not line or line.startswith(b":"):
            continue
        if line.startswith(b"data:"):
            payload = line[5:].strip()
            if payload == b"[DONE]":
                done = True
                break
        elif line.startswith(b"{"):  # a plain JSON body instead of SSE (an error, usually)
            payload = line
        else:
            continue
        try:
            chunk = json.loads(payload.decode("utf-8"))
        except ValueError as exc:
            raise ProviderError(f"unreadable stream chunk: {exc}") from exc
        if isinstance(chunk, dict):
            yield chunk
    if not done and not stop.is_set():
        raise ProviderError("connection failed: stream ended without [DONE]")


@dataclass
class _Request:
    body: dict[str, Any]
    default_cap: bool  # True when max_tokens is OUR default, not something the user set


class ProviderClient(ABC):
    @abstractmethod
    def complete(self, *, model: str, messages: list[dict[str, Any]],
                 tools: Optional[list[dict[str, Any]]] = None, **settings: Any) -> AssistantTurn: ...

    def stream(self, *, model: str, messages: list[dict[str, Any]],
               tools: Optional[list[dict[str, Any]]] = None,
               on_delta: Optional[Callable[[str, str], None]] = None,
               cancel: Optional[threading.Event] = None, **settings: Any) -> AssistantTurn:
        """Default for providers that cannot stream: one blocking call, then one delta."""
        turn = self.complete(model=model, messages=messages, tools=tools, **settings)
        if on_delta and turn.text:
            on_delta("text", turn.text)
        return turn


class OpenAICompatProvider(ProviderClient):
    def __init__(self, base_url: str, api_key: Optional[str] = None, *, timeout: float = 120.0,
                 delays: tuple[float, ...] = RETRY_DELAYS) -> None:
        self.base_url = base_url.rstrip("/")  # so joining "/chat/completions" never gives "//"
        self._api_key = api_key
        self.timeout = timeout  # seconds without any byte before giving up (also the idle limit of a stream)
        self.delays = delays
        self.replay_reasoning = True  # send earlier turns' thinking back (needed by tool-calling chains)
        self.reasoning_window: Optional[int] = None  # None = all of it; N = the last N turns; 0 = none
        self.debug: Optional[Callable[[str, Any], None]] = None  # called with ("request"|"response", data)
        self.default_max_tokens = DEFAULT_MAX_TOKENS  # 0 = send none (a server refused ours and we learned it)

    @classmethod
    def for_backend(cls, name: str) -> "OpenAICompatProvider":
        base_url, key_env = BACKENDS[name]  # an unknown name is a KeyError
        key = os.environ.get(key_env) if key_env else None
        if key_env and not key:
            raise ProviderError(f"{key_env} is not set (put it in .env or the environment)")
        provider = cls(base_url, key)
        provider.default_max_tokens = BACKEND_MAX_TOKENS.get(name, DEFAULT_MAX_TOKENS)
        return provider

    # -- request ------------------------------------------------------------------------

    def _build(self, model: str, messages: list[dict[str, Any]], tools: Optional[list[dict[str, Any]]],
               settings: dict[str, Any], *, stream: bool) -> _Request:
        body: dict[str, Any] = {
            "model": model,
            "messages": wire_messages(messages, replay_reasoning=self.replay_reasoning,
                                      keep_last=self.reasoning_window),
        }
        if tools:
            body["tools"] = tools
        for key in ("temperature", "max_tokens", "reasoning_effort"):  # only what has a value
            if settings.get(key) is not None:
                body[key] = settings[key]
        default_cap = "max_tokens" not in body
        if default_cap and self.default_max_tokens:
            body["max_tokens"] = self.default_max_tokens
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return _Request(body, default_cap)

    @staticmethod
    def _fix(req: _Request, error_text: str) -> bool:
        """The server answered 400 and named a parameter we sent: drop it so the next request
        runs on the server's defaults. True when something was dropped. A 400 that names none
        of these is a real error and is left alone (reference `openai_provider.py:140-163`)."""
        text = error_text.lower()
        body = req.body
        if "reasoning_effort" in body and "effort" in text:
            body.pop("reasoning_effort")
        elif "stream_options" in body and "stream_options" in text:
            body.pop("stream_options")
        elif req.default_cap and "max_tokens" in body and "max_tokens" in text:
            body.pop("max_tokens")  # our default ceiling is above this model's limit
            req.default_cap = False
        else:
            return False
        return True

    @staticmethod
    def _sleep(seconds: float, cancel: Optional[threading.Event]) -> None:
        if cancel is not None:
            if cancel.wait(seconds):
                raise StreamCancelled(AssistantTurn())
        else:
            time.sleep(seconds)

    def _open(self, payload: bytes, cancel: Optional[threading.Event]) -> Any:
        """POST and return the open response. Transient failures (429, 5xx, network) are retried
        after RETRY_DELAYS; a 4xx is the same request failing the same way, so it is not."""
        headers = {"Content-Type": "application/json", "User-Agent": "minicw/0.1",
                   "Accept": "application/json, text/event-stream"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        last: Optional[ProviderError] = None
        for attempt in range(len(self.delays) + 1):
            req = urllib.request.Request(self.base_url + "/chat/completions", payload, headers)
            wait: Optional[float] = None
            try:
                return urllib.request.urlopen(req, timeout=self.timeout)
            except urllib.error.HTTPError as exc:
                text = exc.read().decode("utf-8", "replace")
                last = ProviderError(f"HTTP {exc.code}: {text[:300]}", exc.code, text)
                wait = _retry_after(exc.headers.get("Retry-After") if exc.headers else None)
            except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError) as exc:
                last = ProviderError(f"connection failed: {type(exc).__name__}: {exc}")
            if not _retryable(last) or attempt >= len(self.delays):
                raise last
            self._sleep(wait if wait is not None else self.delays[attempt], cancel)
        raise last  # pragma: no cover

    def _send(self, req: _Request, cancel: Optional[threading.Event]) -> Any:
        """_open plus the "drop the parameter the server refused" loop (at most 3 fixes)."""
        fixes = 0
        while True:
            if self.debug:
                self.debug("request", req.body)
            try:
                return self._open(json.dumps(req.body).encode("utf-8"), cancel)
            except ProviderError as exc:
                if exc.status != 400 or fixes >= 3 or not self._fix(req, exc.body):
                    raise
                if "max_tokens" not in req.body and req.default_cap is False:
                    self.default_max_tokens = 0  # remember: do not offer our ceiling to this server again
                fixes += 1

    # -- complete -----------------------------------------------------------------------

    def complete(self, *, model: str, messages: list[dict[str, Any]],
                 tools: Optional[list[dict[str, Any]]] = None, **settings: Any) -> AssistantTurn:
        req = self._build(model, messages, tools, settings, stream=False)
        resp = self._send(req, None)
        try:
            with resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (OSError, http.client.HTTPException, ValueError) as exc:
            raise ProviderError(f"connection failed: {type(exc).__name__}: {exc}") from exc
        if self.debug:
            self.debug("response", data)
        # Some gateways answer HTTP 200 with {"error": ...}; without this it would look like an
        # empty reply from the model.
        if not data.get("choices") and data.get("error"):
            raise ProviderError(f"provider error in a 200 reply: {str(data['error'])[:300]}", 200, json.dumps(data))
        return parse_response(data)

    # -- stream -------------------------------------------------------------------------

    def stream(self, *, model: str, messages: list[dict[str, Any]],
               tools: Optional[list[dict[str, Any]]] = None,
               on_delta: Optional[Callable[[str, str], None]] = None,
               cancel: Optional[threading.Event] = None, **settings: Any) -> AssistantTurn:
        req = self._build(model, messages, tools, settings, stream=True)
        events: queue.Queue = queue.Queue()
        stop = threading.Event()  # tells the worker to stop reading (set on cancel, error or end)

        def worker() -> None:
            resp = None
            try:
                resp = self._send(req, cancel)
                for chunk in iter_sse(resp, stop):
                    events.put(("chunk", chunk))
                events.put(("end", None))
            except (ProviderError, StreamCancelled) as exc:
                events.put(("error", exc))
            except BaseException as exc:  # a socket error in the middle of the body, for example
                events.put(("error", ProviderError(f"connection failed: {type(exc).__name__}: {exc}")))
            finally:
                if resp is not None:
                    try:
                        resp.close()  # closed by the thread that reads it, never by the caller
                    except Exception:  # noqa: BLE001
                        pass

        threading.Thread(target=worker, daemon=True, name="minicw-stream").start()
        acc = StreamAccumulator(on_delta)
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    raise StreamCancelled(acc.partial())
                try:
                    kind, value = events.get(timeout=0.1)
                except queue.Empty:
                    continue
                if kind == "chunk":
                    acc.feed(value)
                elif kind == "end":
                    break
                else:
                    if isinstance(value, StreamCancelled):
                        raise StreamCancelled(acc.partial()) from None
                    raise value
        finally:
            stop.set()
        turn = acc.finish()
        if self.debug:
            self.debug("response", {"text": turn.text, "tool_calls": [(c.name, c.arguments) for c in turn.tool_calls],
                                    "finish_reason": turn.finish_reason})
        return turn


# -- test double ---------------------------------------------------------------------------


@dataclass
class Hang:
    """Script item for ScriptedProvider: the model "talks" `text`, then stalls until cancelled."""
    text: str = ""


class ScriptedProvider(ProviderClient):
    """Replays a fixed list of AssistantTurns (or Exceptions, or Hang), and records what it was
    sent so tests can assert on the outbound messages. The only fake boundary the tests need."""

    def __init__(self, turns: list[Any]) -> None:
        self._turns = list(turns)
        self.calls: list[dict[str, Any]] = []

    def _next(self, model: str, messages: list[dict[str, Any]], tools: Any) -> Any:
        self.calls.append({"model": model, "messages": list(messages), "tools": tools})
        if not self._turns:
            raise ProviderError("ScriptedProvider ran out of turns")
        item = self._turns.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def complete(self, *, model, messages, tools=None, **settings) -> AssistantTurn:
        item = self._next(model, messages, tools)
        if isinstance(item, Hang):
            raise ProviderError("Hang is for stream() only")
        return item

    def stream(self, *, model, messages, tools=None, on_delta=None, cancel=None, **settings) -> AssistantTurn:
        item = self._next(model, messages, tools)
        if isinstance(item, Hang):
            if on_delta and item.text:
                on_delta("text", item.text)
            while cancel is None or not cancel.is_set():
                time.sleep(0.02)
            raise StreamCancelled(AssistantTurn(text=item.text or None, finish_reason="interrupted"))
        if on_delta:
            if item.reasoning:
                on_delta("reasoning", item.reasoning)
            if item.text:
                on_delta("text", item.text)
        return item
