"""Shared helpers: a scripted provider, an engine in a temp workspace, tiny turn builders."""

from __future__ import annotations

import pytest

from minicw.app import build_engine
from minicw.core.config import Settings
from minicw.core.models import AssistantTurn, TokenUsage, ToolCall
from minicw.core.permissions import allow_all
from minicw.core.provider import ScriptedProvider
from minicw.core.sessions import Session


def call(tool_name: str, call_id: str = "c1", /, **arguments) -> ToolCall:
    return ToolCall(id=call_id, name=tool_name, arguments=arguments)


def turn(text=None, calls=(), finish="stop", usage=None) -> AssistantTurn:
    return AssistantTurn(text=text, tool_calls=list(calls),
                         finish_reason="tool_calls" if calls else finish,
                         usage=usage or TokenUsage(input=100, output=20))


@pytest.fixture
def state(tmp_path):
    d = tmp_path / "state"
    d.mkdir()
    return d


@pytest.fixture
def ws(tmp_path):
    d = tmp_path / "ws"
    d.mkdir()
    return d


@pytest.fixture
def make(state, ws, monkeypatch):
    """make(turns, mode=..., approver=...) -> (engine, provider, session, events)."""
    monkeypatch.setenv("MINICW_STATE_DIR", str(state))

    def _make(turns, mode="auto", approver=allow_all, ask_user=None, plan_approver=None,
              session=None, messages=None, context_window=128_000):
        provider = ScriptedProvider(turns)
        settings = Settings(backend="ollama", model="fake", mode=mode, context_window=context_window).resolved()
        if session is None:
            session = Session.create(state, workspace=str(ws), backend="ollama", model="fake", mode=mode)
        events: list[dict] = []
        engine = build_engine(settings=settings, workspace=ws, state=state, provider=provider, session=session,
                              messages=messages if messages is not None else [], on_event=events.append,
                              approver=approver, ask_user=ask_user, plan_approver=plan_approver)
        return engine, provider, session, events
    return _make
