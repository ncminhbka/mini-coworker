"""Web guard and tools, the MCP client against a fake server, Gmail and Calendar against fakes."""

from __future__ import annotations

import base64
import json
import sys
import threading
import time
from email import message_from_bytes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from conftest import call, turn
from minicw.core.permissions import Answer
from minicw.core.risk import Risk
from minicw.integrations.google import make_google_tools
from minicw.integrations.google.calendar import make_calendar_tools
from minicw.integrations.google.gmail import make_gmail_tools
from minicw.integrations.mcp import (McpError, McpManager, ServerConfig, StdioClient, load_config, make_tool_specs,
                                     tool_name)
from minicw.tools import web

FAKE = str(Path(__file__).with_name("fake_mcp_server.py"))


# -- web -----------------------------------------------------------------------------------


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/", "http://localhost:8080/x", "http://169.254.169.254/latest/meta-data/",
    "http://10.0.0.5/", "http://192.168.1.1/", "http://100.64.1.1/", "http://[::1]/", "http://[::ffff:127.0.0.1]/",
    "http://0.0.0.0/", "file:///etc/passwd", "ftp://example.com/", "http:///nohost",
])
def test_the_guard_refuses_internal_and_non_web_urls(url):
    with pytest.raises(ValueError):
        web.check_url(url)


def test_html_becomes_readable_text():
    html = ("<html><head><title> My  page </title><style>p{color:red}</style></head><body><h1>Hello</h1>"
            "<script>alert(1)</script><p>one <b>two</b></p><ul><li>a</li><li>b</li></ul>&amp; done</body></html>")
    title, text = web.html_to_text(html)
    assert title == "My page"
    assert "alert" not in text and "color" not in text
    assert text.splitlines()[0] == "Hello" and "one two" in text and "& done" in text


class _Site(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/page":
            body = "<html><title>T</title><body><p>" + "word " * 5000 + "</p></body></html>"
            self._reply(200, "text/html; charset=utf-8", body.encode())
        elif self.path == "/to-internal":
            self.send_response(302)
            self.send_header("Location", "http://internal.example/secret")
            self.end_headers()
        elif self.path == "/loop":
            self.send_response(302)
            self.send_header("Location", "/loop")
            self.end_headers()
        elif self.path == "/image":
            self._reply(200, "image/png", b"\x89PNG....")
        else:
            self._reply(404, "text/plain", b"nope")

    def _reply(self, status, ctype, body):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def site(monkeypatch):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Site)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def allow_local_only(url):  # the real guard blocks 127.0.0.1; here "internal.example" plays the private host
        if "internal.example" in url:
            raise ValueError("internal.example resolves to a private or internal address; not fetched")

    monkeypatch.setattr(web, "check_url", allow_local_only)
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_fetch_returns_text_title_and_truncates(site):
    out = web.fetch_url(site + "/page", max_chars=1000)
    assert out["title"] == "T" and out["truncated"] is True and len(out["text"]) == 1000


def test_a_redirect_to_an_internal_address_is_refused_at_the_hop(site):
    out = web.fetch_url(site + "/to-internal")
    assert "error" in out and "private or internal" in out["error"]


def test_redirect_loops_and_binary_pages_and_404(site):
    assert "redirects" in web.fetch_url(site + "/loop")["error"]
    assert "not a text page" in web.fetch_url(site + "/image")["error"]
    assert "404" in web.fetch_url(site + "/missing")["error"]


def test_web_tools_through_the_engine(make):
    seen = []

    def fake_search(query, n):
        seen.append((query, n))
        return [{"title": "T", "url": "https://example.org", "snippet": "S"}]

    from minicw.tools.registry import ToolRegistry

    engine, _p, _s, _e = make([turn(calls=[call("web_search", "w1", query="minicw", max_results=99)]), turn("ok")],
                              mode="auto")
    engine.registry.register(web.make_web_tools(fake_search)[0])  # replace the real search by the fake
    engine.run("search")
    assert seen == [("minicw", 10)]  # max_results is clamped to 10
    tool = json.loads([m for m in engine.messages if m["role"] == "tool"][0]["content"])
    assert tool["results"][0]["url"] == "https://example.org"


def test_web_tools_are_egress_and_ask_in_ask_mode(make):
    asked = []
    engine, _p, _s, _e = make([turn(calls=[call("web_fetch", "f1", url="http://127.0.0.1/")]), turn("ok")], mode="ask",
                              approver=lambda t, a, d: asked.append(t) or Answer.ONCE)
    engine.run("fetch")
    assert asked == ["web_fetch"]
    assert "private or internal" in [m for m in engine.messages if m["role"] == "tool"][0]["content"]  # the real guard


# -- MCP -----------------------------------------------------------------------------------


@pytest.fixture
def server(tmp_path, monkeypatch):
    log = tmp_path / "fake.log"
    monkeypatch.setenv("FAKE_MCP_LOG", str(log))
    clients = []

    def start(**kw) -> StdioClient:
        client = StdioClient(ServerConfig(name="fake", command=sys.executable, args=[FAKE], **kw), tmp_path / "mcp.log")
        client.start()
        client.initialize(timeout=20)
        client.tools = client.list_tools()
        clients.append(client)
        return client

    start.log = log
    yield start
    for c in clients:
        c.close()


def test_handshake_tools_and_calls(server):
    client = server()
    assert [t["name"] for t in client.tools] == ["echo", "boom", "slow", "die", "bad name!"]  # the junk line was ignored
    assert client.call_tool("echo", {"text": "hi"}) == {"result": "hi\n[image]"}
    assert client.call_tool("boom", {}) == {"error": "kaboom"}
    with pytest.raises(McpError, match="unknown tool"):
        client.call_tool("nope", {})


def test_requests_from_the_server_get_a_method_not_supported_reply(server):
    client = server()
    client.call_tool("echo", {"text": "x"})
    time.sleep(0.3)
    assert "client replied to srv1" in server.log.read_text() and "-32601" in server.log.read_text()


def test_a_timeout_does_not_poison_later_calls(server):
    client = server()
    with pytest.raises(McpError, match="no answer"):
        client.call_tool("slow", {"seconds": 1.5}, timeout=0.4)
    time.sleep(1.5)  # the late reply arrives and is dropped
    assert client.call_tool("echo", {"text": "still fine"})["result"].startswith("still fine")


def test_a_server_that_dies_mid_call_is_reported(server):
    client = server()
    with pytest.raises(McpError, match="stopped|not running"):
        client.call_tool("die", {})
    assert not client.alive()


def test_tool_specs_names_risk_and_closure(server):
    client = server()
    specs = {s.name: s for s in make_tool_specs(client, set())}
    assert set(specs) == {"mcp__fake__echo", "mcp__fake__boom", "mcp__fake__slow", "mcp__fake__die", "mcp__fake__bad_name_"}
    echo = specs["mcp__fake__echo"]
    assert echo.risk == Risk.EXTERNAL and echo.parameters["required"] == ["text"]
    assert echo.description.startswith("[MCP server fake]")
    # a model cannot redirect the call to another tool by passing a '_remote' (or any other) argument
    assert echo.func(text="x", _remote="boom") == {"result": "x\n[image]"}
    assert '"_remote": "boom"' in server.log.read_text()  # it went to the server as an ordinary argument of echo


def test_tool_names_are_clean_unique_and_at_most_64_characters():
    taken: set[str] = set()
    a = tool_name("my server", "do.it", taken)
    b = tool_name("my server", "do.it", taken)
    long = tool_name("s", "x" * 100, taken)
    assert a == "mcp__my_server__do_it" and b == "mcp__my_server__do_it_2" and len(long) == 64


def test_config_expansion_filters_and_problems(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "abc")
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {
        "ok": {"command": "node", "args": ["srv.js", "--key=${MY_TOKEN}"], "env": {"K": "${MY_TOKEN}"},
               "include_tools": ["a"], "exclude_tools": ["b"]},
        "off": {"command": "node", "enabled": False},
        "remote": {"url": "https://example.com/mcp"},
        "broken": {"args": []},
    }}))
    servers, problems = load_config(cfg)
    assert [s.name for s in servers] == ["ok"]
    assert servers[0].args == ["srv.js", "--key=abc"] and servers[0].env == {"K": "abc"}
    assert servers[0].include_tools == ["a"] and servers[0].exclude_tools == ["b"]
    assert len(problems) == 2 and any("HTTP" in p for p in problems) and any("command" in p for p in problems)
    assert load_config(tmp_path / "missing.json") == ([], [])
    (tmp_path / "bad.json").write_text("{not json")
    assert load_config(tmp_path / "bad.json")[1]


def test_manager_starts_servers_reports_failures_and_feeds_the_engine(make, state, monkeypatch):
    monkeypatch.setenv("FAKE_MCP_LOG", str(state / "fake.log"))
    (state / "mcp.json").write_text(json.dumps({"mcpServers": {
        "fake": {"command": sys.executable, "args": [FAKE], "exclude_tools": ["die", "slow"]},
        "ghost": {"command": "definitely-not-a-program-xyz"},
    }}))
    manager = McpManager(state)
    manager.start(wait=30)
    try:
        assert manager.status["fake"].state == "ready" and manager.status["ghost"].state == "failed"
        assert "not found" in manager.status["ghost"].detail
        names = {s.name for s in manager.tool_specs()}
        assert names == {"mcp__fake__echo", "mcp__fake__boom", "mcp__fake__bad_name_"}

        from minicw.app import build_engine
        from minicw.core.config import Settings
        from minicw.core.provider import ScriptedProvider
        from minicw.core.sessions import Session

        asked = []
        session = Session.create(state, workspace=str(state), backend="ollama", model="fake", mode="ask")
        engine = build_engine(
            settings=Settings(backend="ollama", model="fake", mode="ask").resolved(), workspace=state, state=state,
            provider=ScriptedProvider([turn(calls=[call("mcp__fake__echo", "m1", text="from the model")]), turn("done")]),
            session=session, messages=[], on_event=lambda e: None,
            approver=lambda t, a, d: asked.append(t) or Answer.ONCE, extra_tools=manager.tool_specs())
        engine.run("use the server")
        assert asked == ["mcp__fake__echo"]  # EXTERNAL: it asks
        assert "from the model" in [m for m in engine.messages if m["role"] == "tool"][0]["content"]
    finally:
        manager.close()


# -- Google, with fake services --------------------------------------------------------------


def chain(**results):
    """A fake API client: svc.<a>().<b>().<verb>(...).execute() returns the given value."""
    svc = MagicMock()
    for path, value in results.items():
        node = svc
        *parents, verb = path.split(".")
        for p in parents:
            node = getattr(node, p).return_value
        getattr(node, verb).return_value.execute.return_value = value
    return svc


def b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def tools_by_name(specs):
    return {s.name: s for s in specs}


def test_gmail_search_and_read():
    svc = chain(**{"users.messages.list": {"messages": [{"id": "m1"}]},
                   "users.messages.get": {"id": "m1", "snippet": "hi there", "payload": {
                       "headers": [{"name": "From", "value": "ann@x.org"}, {"name": "Subject", "value": "Lunch?"},
                                   {"name": "Date", "value": "Mon"}, {"name": "To", "value": "me@x.org"}],
                       "mimeType": "multipart/alternative", "parts": [
                           {"mimeType": "text/html", "body": {"data": b64("<p>html</p>")}},
                           {"mimeType": "text/plain", "body": {"data": b64("plain body Tiếng Việt")}}]}}})
    t = tools_by_name(make_gmail_tools(lambda: svc))
    found = t["gmail_search_messages"].func(query="from:ann", max_results=500)
    assert found["messages"] == [{"id": "m1", "from": "ann@x.org", "subject": "Lunch?", "date": "Mon", "snippet": "hi there"}]
    assert svc.users().messages().list.call_args.kwargs["maxResults"] == 20  # clamped
    msg = t["gmail_get_message"].func(message_id="m1")
    assert msg["body"] == "plain body Tiếng Việt" and msg["subject"] == "Lunch?" and msg["to"] == "me@x.org"
    assert t["gmail_search_messages"].risk == Risk.EGRESS


def test_gmail_send_builds_a_real_message_and_refuses_header_injection():
    svc = chain(**{"users.messages.send": {"id": "sent1"}})
    send = tools_by_name(make_gmail_tools(lambda: svc))["gmail_send_email"]
    assert send.risk == Risk.EXTERNAL
    assert send.func(to="bob@x.org", subject="Hi", body="Nội dung", cc="c@x.org") == {"sent": True, "id": "sent1", "to": "bob@x.org"}
    raw = svc.users().messages().send.call_args.kwargs["body"]["raw"]
    mail = message_from_bytes(base64.urlsafe_b64decode(raw))
    assert mail["To"] == "bob@x.org" and mail["Subject"] == "Hi" and mail["Cc"] == "c@x.org"
    assert "Nội dung" in mail.get_payload(decode=True).decode("utf-8")
    bad = send.func(to="bob@x.org", subject="Hi\nBcc: evil@x.org", body="x")
    assert "error" in bad and svc.users().messages().send.call_count == 1  # nothing more was sent


def test_calendar_tools():
    svc = chain(**{"events.list": {"items": [{"id": "e1", "summary": "Standup", "start": {"dateTime": "2026-11-05T09:00:00+07:00"},
                                              "end": {"dateTime": "2026-11-05T09:15:00+07:00"}, "attendees": [{"email": "a@x"}]}]},
                   "events.insert": {"id": "e2", "summary": "Trip", "start": {"date": "2026-11-10"}, "end": {"date": "2026-11-12"}},
                   "events.patch": {"id": "e1", "summary": "Standup (moved)", "start": {}, "end": {}},
                   "events.delete": ""})
    t = tools_by_name(make_calendar_tools(lambda: svc))
    events = t["gcal_list_events"].func()["events"]
    assert events[0]["summary"] == "Standup" and events[0]["attendees"] == ["a@x"]
    assert "timeMin" in svc.events().list.call_args.kwargs  # defaults to now
    created = t["gcal_create_event"].func(summary="Trip", start="2026-11-10", end="2026-11-12", attendees=["b@x"])
    assert created["start"] == "2026-11-10"
    body = svc.events().insert.call_args.kwargs["body"]
    assert body["start"] == {"date": "2026-11-10"} and body["attendees"] == [{"email": "b@x"}]
    t["gcal_create_event"].func(summary="Call", start="2026-11-05T10:00:00", end="2026-11-05T11:00:00", timezone_name="Asia/Ho_Chi_Minh")
    assert svc.events().insert.call_args.kwargs["body"]["start"] == {"dateTime": "2026-11-05T10:00:00", "timeZone": "Asia/Ho_Chi_Minh"}
    assert "error" in t["gcal_update_event"].func(event_id="e1")  # nothing to change
    assert t["gcal_update_event"].func(event_id="e1", summary="Standup (moved)")["summary"] == "Standup (moved)"
    assert t["gcal_delete_event"].func(event_id="e1") == {"deleted": True, "id": "e1"}
    assert [n for n, s in t.items() if s.risk == Risk.EXTERNAL] == ["gcal_create_event", "gcal_update_event", "gcal_delete_event"]


def test_google_errors_come_back_as_results_not_crashes():
    def broken():
        raise RuntimeError("not signed in to Google: run /google login")

    out = tools_by_name(make_calendar_tools(broken))["gcal_list_events"].func()
    assert out == {"error": "RuntimeError: not signed in to Google: run /google login"}


def test_google_tools_only_exist_when_there_is_an_oauth_client(state):
    assert make_google_tools(state) == []  # no credentials.json


def test_config_written_by_windows_powershell_with_a_bom_is_read(tmp_path):
    cfg = tmp_path / "mcp.json"
    cfg.write_bytes("﻿".encode("utf-8") + json.dumps({"mcpServers": {"x": {"command": "node"}}}).encode())
    assert [s.name for s in load_config(cfg)[0]] == ["x"]
