"""The agent loop, for a conversation of many turns that can be stopped with Ctrl-C.

One `run(text)` is one turn: call the model (streaming), append its reply, run its tool calls
through the permission engine, append each result, repeat until the model answers without
tool calls, a limit trips, or the user interrupts. The engine lives for the whole session;
every user line is another `run()` appended to the same `messages`.

From mcw `engine.py`, which kept these invariants because weak models trip them:
* every tool call gets exactly one tool message with the matching id (errors, denials and
  interrupted calls included), so a provider never sees an orphan;
* a call whose arguments never parsed is not executed; the model is told what went wrong;
* a reply cut off at the output limit is not "done": the model is asked to continue;
* text that looks like a tool call but did not parse ends the turn as an error;
* compaction only changes the outbound copy, never `self.messages`.

New for minicw:
* `cancel` (a threading.Event) is set by the SIGINT handler during a turn. The engine looks at
  it between chunks, before each tool and while a tool runs. A second Ctrl-C re-raises
  KeyboardInterrupt, so a demo that hangs can still be killed.
* Blocking work (tools, the compaction summary) runs in a worker thread so the main thread can
  keep watching `cancel`. A tool that ignores cancel is abandoned after a short grace period.
* Every message is passed to `on_message` the moment it is appended (the session writes it to
  disk), so a crash loses at most the message in flight.
* Calls that were pending at an interrupt are closed with an error result IMMEDIATELY, so an
  orphan can only ever sit at the tail of the history, and `repair()` only appends.
* The plan-mode notice is added to the outbound copy of the last user message, not stored.
"""

from __future__ import annotations

import json
import re
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from . import compaction
from .audit import AuditLog
from .permissions import Answer, Approver, Decision, Mode, PermissionEngine, Verdict, deny_all
from .prompt import PLAN_NOTICE
from .provider import ProviderClient, ProviderError, StreamCancelled
from .models import RAW_KEY, AssistantTurn, ToolCall, TokenUsage
from ..tools.registry import ToolRegistry

MAX_TRUNCATION_CONTINUATIONS = 3
TOOL_RESULT_MAX_CHARS = 10_000
CANCEL_GRACE = 2.0  # seconds a running tool gets to notice cancel before it is abandoned

TRUNCATION_NUDGE = (
    "Your last reply was cut off at the output limit before you did anything. Do not repeat "
    "the long reasoning: make the tool call now, or give the answer in fewer words."
)
# What the provider sees in place of a reply that was cut off before any action. The transcript
# keeps the original; resending partial thinking every turn would cost up to max_tokens each time.
TRUNCATION_STUB = "(reply cut off at the output-token limit before any action)"
MANGLED_PREVIEW_CHARS = 200
CUT_OFF_ARGS_ERROR = (
    "Your tool-call arguments were cut off by the output-token limit before they finished, so "
    "the call was NOT run. Produce the same content in smaller pieces: several calls that each "
    "write or append one section, each well under the limit. Do not retry the identical call."
)
BAD_JSON_ARGS_ERROR = (
    "The arguments for this call were not valid JSON, so it was NOT run. `_raw` is not a "
    "parameter: it is the unparsed text of the failed call. Resend the call with a single "
    "valid JSON object using the tool's declared parameters."
)
INTERRUPTED_ERROR = "interrupted by the user before this call produced a result; it may or may not have run"
NOT_RUN_ERROR = "interrupted by the user; this call was not run"

# Tools the engine answers itself, on the main thread, because they talk to the person.
INTERACTIVE_TOOLS = ("ask_user", "propose_plan")


@dataclass
class RunResult:
    # completed | interrupted | max_iterations_exceeded | truncated | error
    status: str
    text: str = ""
    iterations: int = 0
    error: Optional[str] = None


@dataclass
class EngineConfig:
    max_iterations: int = 100
    context_window: int = compaction.DEFAULT_CONTEXT_WINDOW
    tool_result_max_chars: int = TOOL_RESULT_MAX_CHARS
    settings: dict[str, Any] = field(default_factory=dict)  # temperature, max_tokens, reasoning_effort


class _Interrupted(Exception):
    """Raised inside the engine when cancel is seen; never leaves `run()`."""


class Engine:
    def __init__(
        self,
        provider: ProviderClient,
        model: str,
        registry: ToolRegistry,
        permissions: PermissionEngine,
        *,
        system_prompt: str,
        cancel: Optional[threading.Event] = None,
        approver: Approver = deny_all,
        on_event: Optional[Callable[[dict[str, Any]], None]] = None,
        on_message: Optional[Callable[[dict[str, Any]], None]] = None,
        on_rewrite: Optional[Callable[[list[dict[str, Any]]], None]] = None,
        on_checkpoint: Optional[Callable[[], None]] = None,
        audit: Optional[AuditLog] = None,
        messages: Optional[list[dict[str, Any]]] = None,
        compaction_state: Optional[compaction.CompactionState] = None,
        usage: Optional[dict[str, int]] = None,
        config: Optional[EngineConfig] = None,
        ask_user: Optional[Callable[[str, list[str]], Optional[str]]] = None,
        plan_approver: Optional[Callable[[str], tuple[bool, str]]] = None,
        limit_result: Optional[Callable[[ToolCall, str], str]] = None,
    ) -> None:
        self.provider, self.model = provider, model
        self.registry, self.permissions = registry, permissions
        self.system_prompt = system_prompt
        self.cancel = cancel or threading.Event()
        self.approver = approver
        self.on_event = on_event or (lambda _e: None)
        self.on_message = on_message or (lambda _m: None)  # the session writes each message here
        self.on_rewrite = on_rewrite or (lambda _ms: None)  # atomic rewrite, for the rare middle repair
        self.on_checkpoint = on_checkpoint or (lambda: None)  # save metadata (grants, compaction, usage)
        self.audit = audit
        self.messages: list[dict[str, Any]] = messages if messages is not None else []
        self.compaction_state = compaction_state
        self.usage = usage or {"input": 0, "output": 0, "cache_read": 0, "turns": 0}
        self.config = config or EngineConfig()
        self.ask_user_cb, self.plan_approver = ask_user, plan_approver
        self.limit_result = limit_result
        self._last_context_tokens = 0  # provider's own count of the last request
        self._usage_index = len(self.messages)  # messages from here on came after that count
        self.session_id = ""
        # Set by app.build_engine, used by the UI (/persona, /memory, /skills) and the snapshot.
        self.persona: Any = None
        self.missing_tools: list[str] = []
        self.memory: Any = None
        self.skills: Any = None
        if self.compaction_state is not None and not self.compaction_state.valid_for(self.messages):
            self.compaction_state = None  # a saved state that does not fit the history is not trusted

    # ------------------------------------------------------------------------------------
    # history
    # ------------------------------------------------------------------------------------

    def _add(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        self.on_message(message)

    def pending_calls(self) -> list[dict[str, Any]]:
        """Tool calls of the LAST assistant message that have no result yet."""
        answered = {m.get("tool_call_id") for m in self.messages if m.get("role") == "tool"}
        for m in reversed(self.messages):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                return [tc for tc in m["tool_calls"] if tc.get("id") not in answered]
            if m.get("role") == "user":
                break
        return []

    def close_pending(self, error: str = INTERRUPTED_ERROR) -> None:
        """Give every unanswered call a result NOW. Providers reject a history with an orphan
        (an assistant tool call that no tool message answers), and one orphan would poison the
        whole session."""
        for tc in self.pending_calls():
            self._add(_stub(tc, error))

    def repair(self) -> int:
        """Close orphans left by an older crash. Normally they can only be at the tail, and this
        just appends. An orphan in the middle (a damaged file) gets its stub inserted right after
        its assistant message, the compaction boundary is shifted, and the file is rewritten.
        Returns how many stubs were added."""
        answered = {m.get("tool_call_id") for m in self.messages if m.get("role") == "tool"}
        added, middle = 0, False
        i = 0
        while i < len(self.messages):
            msg = self.messages[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                stubs = [_stub(tc) for tc in msg["tool_calls"] if tc.get("id") and tc["id"] not in answered]
                if stubs:
                    at = i + 1
                    while at < len(self.messages) and self.messages[at].get("role") == "tool":
                        at += 1  # after the results this message already has
                    if at == len(self.messages):
                        for stub in stubs:
                            self._add(stub)
                    else:
                        middle = True
                        self.messages[at:at] = stubs
                        if self.compaction_state and self.compaction_state.boundary >= at:
                            self.compaction_state.boundary += len(stubs)
                    answered.update(stub["tool_call_id"] for stub in stubs)
                    added += len(stubs)
            i += 1
        if middle:
            self.on_rewrite(self.messages)
        return added

    # ------------------------------------------------------------------------------------
    # what is sent
    # ------------------------------------------------------------------------------------

    def _outbound(self) -> list[dict[str, Any]]:
        """Messages for the next request: system prompt, the history through compaction (with
        cut-off replies replaced by a stub), and the plan notice on the last user message."""
        history = [_stub_if_cut_off(m) for m in compaction.apply_to_outbound(self.messages, self.compaction_state)]
        if self.permissions.mode == Mode.PLAN:
            for i in range(len(history) - 1, -1, -1):
                if history[i].get("role") == "user" and isinstance(history[i].get("content"), str):
                    history[i] = {**history[i], "content": history[i]["content"] + "\n\n" + PLAN_NOTICE}
                    break
        return [{"role": "system", "content": self.system_prompt}, *history]

    # ------------------------------------------------------------------------------------
    # compaction
    # ------------------------------------------------------------------------------------

    def _trigger(self) -> int:
        return compaction.trigger_tokens(self.config.context_window)

    def _compaction_due(self) -> bool:
        # The provider's own count of the last request plus an estimate of what was added since;
        # the chars/4 estimate of everything only before the first reply or right after a compaction.
        if self._last_context_tokens:
            used = self._last_context_tokens + compaction.estimate_tokens(self.messages[self._usage_index:])
        else:
            used = compaction.estimate_tokens(self._outbound())
        return used >= self._trigger()

    def _compact(self, *, force: bool = False) -> bool:
        def work() -> Optional[compaction.CompactionState]:
            return compaction.compact(self.provider, self.model, self.messages, self.compaction_state,
                                      trigger=self._trigger(), on_usage=self._count_usage, force=force)

        state = self._blocking(work)
        if state is None:
            return False
        self.compaction_state = state
        self._last_context_tokens = 0  # the old figure described the pre-compaction prompt
        self.on_event({"type": "compacted", "boundary": state.boundary, "trimmed": state.trimmed})
        self.on_checkpoint()
        return True

    def _count_usage(self, usage: TokenUsage) -> None:
        self.usage["input"] += usage.input
        self.usage["output"] += usage.output
        self.usage["cache_read"] += usage.cache_read
        self.usage["turns"] += 1

    # ------------------------------------------------------------------------------------
    # blocking work in a thread, so cancel stays responsive
    # ------------------------------------------------------------------------------------

    def _blocking(self, fn: Callable[[], Any]) -> Any:
        """Run fn() in a daemon thread and wait for it in slices of 0.1 s. When cancel is set,
        the work gets CANCEL_GRACE seconds to finish by itself (the shell tool kills its process
        tree and returns); after that the thread is abandoned and _Interrupted is raised."""
        box: dict[str, Any] = {}
        done = threading.Event()

        def target() -> None:
            try:
                box["value"] = fn()
            except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread
                box["error"] = exc
            finally:
                done.set()

        threading.Thread(target=target, daemon=True, name="minicw-work").start()
        while not done.wait(0.1):
            if self.cancel.is_set() and not done.wait(CANCEL_GRACE):
                raise _Interrupted()
        if "error" in box:
            raise box["error"]
        return box.get("value")

    # ------------------------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------------------------

    def run(self, text: str) -> RunResult:
        """One user turn. Never raises for an interrupt: it returns status "interrupted"."""
        with _SigintGuard(self.cancel):
            try:
                self.cancel.clear()
                self.repair()
                return self._run(text)
            except (_Interrupted, StreamCancelled, KeyboardInterrupt):
                return self._interrupted()
            finally:
                self.on_checkpoint()

    def _interrupted(self) -> RunResult:
        self.close_pending()
        self.on_event({"type": "interrupted"})
        self.on_event({"type": "turn_end", "status": "interrupted"})
        return RunResult("interrupted")

    def _check_cancel(self) -> None:
        if self.cancel.is_set():
            raise _Interrupted()

    def _run(self, text: str) -> RunResult:
        self._add({"role": "user", "content": text, "ts": time.time()})
        iterations = continuations = overflow_retries = 0
        while True:
            self._check_cancel()
            if iterations >= self.config.max_iterations:
                return self._end(RunResult("max_iterations_exceeded", iterations=iterations))
            iterations += 1
            if self._compaction_due():
                self._compact()
            self._check_cancel()

            try:
                turn = self._stream_once()
            except StreamCancelled as cancelled:
                self._save_partial(cancelled.partial)
                raise _Interrupted() from None
            except ProviderError as exc:
                # A context overflow the estimate missed goes through compaction. Each pass
                # moves the boundary forward, so a model that keeps overflowing still ends.
                if compaction.is_context_overflow(exc) and overflow_retries < 3 and self._compact(force=True):
                    overflow_retries += 1
                    iterations -= 1
                    continue
                self.on_event({"type": "error", "error": str(exc)})
                return self._end(RunResult("error", iterations=iterations, error=f"{type(exc).__name__}: {exc}"))

            if turn.usage is not None:
                self._last_context_tokens = turn.usage.context_tokens
                self._count_usage(turn.usage)
            else:
                self._last_context_tokens = 0  # a stale count would hide the context's growth
            cut_off = turn.finish_reason == "length"
            raw_texts = _shrink_mangled_calls(turn)
            if not turn.tool_calls:
                cleaned = _DANGLING_TOOL_TAG.sub("", turn.text or "", count=1)
                if cleaned != (turn.text or "") and cleaned.strip() \
                        and not looks_like_unparsed_tool_call(cleaned, self.registry.names()):
                    turn.text = cleaned
            self._add(_assistant_message(turn))
            self._usage_index = len(self.messages)  # context_tokens already counts this reply
            self.on_event({"type": "assistant_message", "text": turn.text,
                           "tool_calls": [tc.name for tc in turn.tool_calls]})

            if not turn.tool_calls:
                if cut_off:
                    if continuations < MAX_TRUNCATION_CONTINUATIONS:
                        continuations += 1
                        self._add({"role": "user", "content": TRUNCATION_NUDGE, "ts": time.time()})
                        continue
                    return self._end(RunResult("truncated", turn.text or "", iterations))
                if looks_like_unparsed_tool_call(turn.text, self.registry.names()):
                    msg = (f"{self.model} replied with a tool call that could not be parsed; "
                           "stopped rather than reporting a half-written call as the answer.")
                    self.on_event({"type": "error", "error": msg})
                    return self._end(RunResult("error", turn.text or "", iterations, msg))
                return self._end(RunResult("completed", turn.text or "", iterations))

            # A call cut off mid-arguments is a truncation too, so it counts against the same limit.
            truncated_call = cut_off and bool(raw_texts)
            continuations = continuations + 1 if truncated_call else 0
            self._handle_calls(turn.tool_calls, cut_off=cut_off)
            if truncated_call and continuations > MAX_TRUNCATION_CONTINUATIONS:
                return self._end(RunResult("truncated", turn.text or "", iterations))

    def _end(self, result: RunResult) -> RunResult:
        self.on_event({"type": "turn_end", "status": result.status})
        return result

    def _stream_once(self) -> AssistantTurn:
        def on_delta(kind: str, piece: str) -> None:
            self.on_event({"type": "assistant_delta" if kind == "text" else "reasoning_delta", "text": piece})

        return self.provider.stream(
            model=self.model, messages=self._outbound(), tools=self.registry.schemas() or None,
            on_delta=on_delta, cancel=self.cancel, **self.config.settings,
        )

    def _save_partial(self, partial: AssistantTurn) -> None:
        """Keep the text that was already shown. Unfinished tool calls are dropped (they have no
        usable arguments), so there is nothing to orphan."""
        if partial.text or partial.reasoning:
            self._add(_assistant_message(partial))

    # ------------------------------------------------------------------------------------
    # tool calls
    # ------------------------------------------------------------------------------------

    def _handle_calls(self, calls: list[ToolCall], *, cut_off: bool) -> None:
        for call in calls:
            if self.cancel.is_set():
                self._tool_message(call, {"error": NOT_RUN_ERROR}, "interrupted")
                continue
            try:
                self._handle_call(call, cut_off=cut_off)
            except (_Interrupted, KeyboardInterrupt):
                # Close this call and all the ones after it, then let run() report the interrupt.
                self.cancel.set()
                self.close_pending()
                raise _Interrupted() from None
        self._check_cancel()

    def _handle_call(self, call: ToolCall, *, cut_off: bool) -> None:
        self.on_event({"type": "tool_proposed", "name": call.name, "arguments": call.arguments, "call_id": call.id})
        if _is_mangled(call):
            self._tool_message(call, {"error": CUT_OFF_ARGS_ERROR if cut_off else BAD_JSON_ARGS_ERROR}, "error")
            return
        spec = self.registry.get(call.name)
        if spec is None:
            self._tool_message(call, {"error": f"unknown tool {call.name!r}", "available": self.registry.names()}, "error")
            return

        # The permission check runs on the main thread; so does the question to the user.
        try:
            decision = self.permissions.evaluate(call.name, call.arguments)
        except Exception as exc:  # noqa: BLE001 - a crash here must not orphan the call
            self._tool_message(call, {"error": f"Not run: the permission check failed ({type(exc).__name__})."}, "denied")
            return
        source = decision.source
        if decision.verdict == Verdict.ASK:
            answer = self._ask_permission(call, decision)
            source = "user" if answer != Answer.DENY else decision.source
            if answer == Answer.DENY:
                self._audit(call, decision, "ask", "user", "denied")
                self._tool_message(call, {"error": "The user declined this action. Do not retry it; "
                                                   "choose another approach or explain what you need."}, "denied")
                return
            if answer == Answer.ALWAYS:
                self.permissions.remember(call.name, call.arguments)
                self.on_checkpoint()
        elif decision.verdict == Verdict.DENY:
            self._audit(call, decision, "deny", decision.source, "denied")
            self._tool_message(call, {"error": f"Not allowed: {decision.reason}."}, "denied")
            return

        if call.name in INTERACTIVE_TOOLS:
            result, status = self._interactive(call)
        else:
            result, status = self._execute(spec.func, call)
        self._audit(call, decision, "allow", source, status)
        self._tool_message(call, result, status)

    def _ask_permission(self, call: ToolCall, decision: Decision) -> Answer:
        """Ask the approver. Ctrl-C or end-of-input while it is waiting counts as "no" AND stops
        the turn."""
        try:
            return self.approver(call.name, call.arguments, decision)
        except (KeyboardInterrupt, EOFError):
            self.cancel.set()
            return Answer.DENY
        except Exception:  # noqa: BLE001 - a broken approver is a "no", not a crash
            return Answer.DENY

    def _execute(self, func: Callable[..., Any], call: ToolCall) -> tuple[Any, str]:
        try:
            result = self._blocking(lambda: func(**call.arguments))
            return result, ("error" if isinstance(result, dict) and "error" in result else "ok")
        except _Interrupted:
            spec = self.registry.get(call.name)
            wrote = spec is not None and spec.risk.value in ("write_local", "exec", "external")
            return {"error": INTERRUPTED_ERROR if wrote else NOT_RUN_ERROR}, "interrupted"
        except TypeError as exc:  # wrong or missing parameters
            return {"error": f"bad arguments for {call.name}: {exc}"}, "error"
        except Exception as exc:  # noqa: BLE001 - a tool failure is information for the model
            return {"error": f"{type(exc).__name__}: {exc}"}, "error"

    def _interactive(self, call: ToolCall) -> tuple[Any, str]:
        args = call.arguments
        if call.name == "ask_user":
            if self.ask_user_cb is None:
                return {"error": "asking the user is not available here; decide by yourself and say what you assumed"}, "error"
            options = [str(o) for o in args.get("options") or []]
            try:
                answer = self.ask_user_cb(str(args.get("question", "")), options)
            except (KeyboardInterrupt, EOFError):
                self.cancel.set()
                raise _Interrupted() from None
            if answer is None:
                return {"skipped": True, "note": "The user skipped the question. Decide yourself and state your assumption."}, "ok"
            return {"answer": answer}, "ok"
        # propose_plan
        if self.permissions.mode != Mode.PLAN:
            return {"error": "propose_plan is only for plan mode"}, "error"
        if self.plan_approver is None:
            return {"error": "plan approval is not available here"}, "error"
        try:
            approved, feedback = self.plan_approver(str(args.get("plan", "")))
        except (KeyboardInterrupt, EOFError):
            self.cancel.set()
            raise _Interrupted() from None
        if not approved:
            return {"approved": False, "feedback": feedback or "The user did not approve the plan; revise it."}, "ok"
        self.permissions.mode = Mode.ASK
        self.on_event({"type": "mode_changed", "mode": Mode.ASK.value})
        self.on_checkpoint()
        return {"approved": True, "note": "plan approved: you may now make the changes"}, "ok"

    def _audit(self, call: ToolCall, decision: Decision, verdict: str, source: str, outcome: str) -> None:
        if self.audit is None:
            return
        spec = self.registry.get(call.name)
        self.audit.record(tool=call.name, arguments=call.arguments, risk=spec.risk.value if spec else None,
                          verdict=verdict, source=source, reason=decision.reason, outcome=outcome)

    def _tool_message(self, call: ToolCall, result: Any, status: str) -> None:
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        limited = self.limit_result(call, text) if self.limit_result else _clip(text, self.config.tool_result_max_chars)
        self._add({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": limited,
                   "status": status, "ts": time.time()})
        self.on_event({"type": "tool_finished", "name": call.name, "status": status, "preview": limited[:160]})

    # ------------------------------------------------------------------------------------
    # state for the session file
    # ------------------------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """What the session file keeps besides the messages."""
        return {"model": self.model, "mode": self.permissions.mode.value, "grants": self.permissions.grants(),
                "persona": self.persona.id if self.persona else "",
                "compaction": self.compaction_state.as_dict() if self.compaction_state else {},
                "usage": dict(self.usage)}


class _SigintGuard:
    """During a turn, Ctrl-C only sets `cancel`; the engine then stops at its next check. A
    second Ctrl-C in the same turn puts Python's default handler back and raises
    KeyboardInterrupt, so a stuck demo can still be killed. Installed only in the main thread
    (signal handlers cannot be set elsewhere); outside a turn the previous handler is restored."""

    def __init__(self, cancel: threading.Event) -> None:
        self.cancel = cancel
        self.previous: Any = None
        self.active = False

    def _handler(self, signum: int, frame: Any) -> None:
        if self.cancel.is_set():
            signal.signal(signal.SIGINT, signal.default_int_handler)
            raise KeyboardInterrupt
        self.cancel.set()

    def __enter__(self) -> "_SigintGuard":
        if threading.current_thread() is threading.main_thread():
            self.previous = signal.signal(signal.SIGINT, self._handler)
            self.active = True
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.active:
            signal.signal(signal.SIGINT, self.previous if self.previous is not None else signal.default_int_handler)


# -- small helpers ---------------------------------------------------------------------------


def _stub(tc: dict[str, Any], error: str = INTERRUPTED_ERROR) -> dict[str, Any]:
    """A tool message that answers an assistant tool call nobody answered."""
    return {"role": "tool", "tool_call_id": tc["id"], "name": (tc.get("function") or {}).get("name", ""),
            "content": json.dumps({"error": error}), "status": "interrupted", "ts": time.time()}


def _clip(text: str, limit: int) -> str:
    """Head + marker + tail. The end of a log or traceback is usually the useful part."""
    if limit <= 0 or len(text) <= limit:
        return text
    head, tail = int(limit * 0.6), int(limit * 0.4)
    return f"{text[:head]}\n[... {len(text) - head - tail} characters omitted ...]\n{text[-tail:]}"


def _is_mangled(call: ToolCall) -> bool:
    """The provider client's fallback for arguments that were not a JSON object."""
    return set(call.arguments or {}) == {RAW_KEY}


def _shrink_mangled_calls(turn: AssistantTurn) -> list[str]:
    """Cut each mangled call's raw text to a preview BEFORE the turn enters history (the whole
    text is half a JSON document that would cost its tokens again every turn); returns the full
    raw texts."""
    raws = []
    for call in turn.tool_calls:
        if _is_mangled(call):
            raw = str(call.arguments[RAW_KEY] or "")
            raws.append(raw)
            if len(raw) > MANGLED_PREVIEW_CHARS:
                call.arguments = {RAW_KEY: f"{raw[:MANGLED_PREVIEW_CHARS]}… [unparsed tool-call text, "
                                           f"{len(raw)} chars, truncated in history]"}
    return raws


def _stub_if_cut_off(msg: dict[str, Any]) -> dict[str, Any]:
    if msg.get("role") == "assistant" and msg.get("finish_reason") == "length" and not msg.get("tool_calls"):
        return {**msg, "content": TRUNCATION_STUB, "reasoning": None}
    return msg


def _assistant_message(turn: AssistantTurn) -> dict[str, Any]:
    # "" not None: strict endpoints reject an assistant message whose content is null and that
    # carries no tool calls, and then the whole history fails with a 400.
    msg: dict[str, Any] = {"role": "assistant", "content": turn.text or "", "ts": time.time()}
    if turn.tool_calls:
        msg["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
            for tc in turn.tool_calls
        ]
    if turn.reasoning:
        msg["reasoning"] = turn.reasoning
        if turn.reasoning_field:
            msg["reasoning_field"] = turn.reasoning_field
    if turn.usage is not None:
        msg["usage"] = turn.usage.as_dict()
    if turn.finish_reason:
        msg["finish_reason"] = turn.finish_reason
    return msg


# Markup that means "this text IS a tool call the endpoint failed to parse". Fenced and inline
# code is removed first: a model explaining tool-call syntax in a ``` block is answering. List
# and fence rule from the reference `providers/openai_provider.py:506-517` @ 59151e3 (MIT).
_LEAKED_TOOL_SYNTAX = (
    "<tool_call>", "</tool_call>", "<function=", "</function>", "<parameter=", "</parameter>",
    "<function_calls>", "<invoke ",
)
_FENCED = re.compile(r"```.*?```|~~~.*?~~~|`[^`\n]*`", re.DOTALL)
# A finished answer with one stray tool tag at its very end is an answer, not a failed call.
_DANGLING_TOOL_TAG = re.compile(r"\s*</?tool_call>\s*$", re.IGNORECASE)


def looks_like_unparsed_tool_call(text: Optional[str], tool_names: list[str]) -> bool:
    """A reply with no parsed calls whose text still reads like one: tool-call markup outside
    code, or a bare JSON object naming a registered tool next to "arguments"/"parameters"."""
    if not text:
        return False
    stripped = _DANGLING_TOOL_TAG.sub("", text, count=1)
    if stripped.strip():
        text = stripped
    prose = _FENCED.sub("", text)
    lowered = prose.lower()
    if any(m in lowered for m in _LEAKED_TOOL_SYNTAX):
        return True
    if '"arguments"' in lowered or '"parameters"' in lowered:
        return any(f'"{name}"' in prose for name in tool_names)
    return False
