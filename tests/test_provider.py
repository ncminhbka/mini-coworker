"""The streaming client against a tiny local HTTP server that speaks server-sent events."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from minicw.core.provider import OpenAICompatProvider, ProviderError, StreamCancelled


def sse(*chunks, done=True, comment=False):
    out = b""
    if comment:
        out += b": keep-alive\n\n"
    for c in chunks:
        out += b"data: " + (c if isinstance(c, bytes) else json.dumps(c).encode()) + b"\n\n"
    if done:
        out += b"data: [DONE]\n\n"
    return out


def delta(**kw):
    return {"choices": [{"index": 0, "delta": kw, "finish_reason": kw.pop("finish_reason", None)}]}


class Server:
    """Replies are consumed in order; each is (status, body, extra) where extra may hold a stall."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.release = threading.Event()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                outer.requests.append(json.loads(self.rfile.read(length)))
                status, body, stall = outer.replies.pop(0)
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream" if status == 200 else "application/json")
                self.end_headers()
                if stall:
                    self.wfile.write(body)
                    self.wfile.flush()
                    outer.release.wait(10)  # hold the connection open without sending more
                else:
                    self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def close(self):
        self.release.set()
        self.httpd.shutdown()


@pytest.fixture
def serve():
    servers = []

    def _serve(replies):
        s = Server(replies)
        servers.append(s)
        return s, OpenAICompatProvider(s.url, "key", delays=(0, 0))

    yield _serve
    for s in servers:
        s.close()


MSG = [{"role": "user", "content": "hi"}]


def test_text_deltas_and_usage(serve):
    server, p = serve([(200, sse(
        delta(content="Hel"), delta(content="lo"), delta(), {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 7, "prompt_cache_hit_tokens": 20}},
        comment=True), False)])
    seen = []
    turn = p.stream(model="m", messages=MSG, on_delta=lambda k, t: seen.append((k, t)))
    assert turn.text == "Hello" and turn.finish_reason == "stop"
    assert seen == [("text", "Hel"), ("text", "lo")]
    assert (turn.usage.input, turn.usage.cache_read, turn.usage.output) == (30, 20, 7)
    assert server.requests[0]["stream"] is True and server.requests[0]["stream_options"] == {"include_usage": True}


def test_reasoning_field_is_remembered(serve):
    _s, p = serve([(200, sse(delta(reasoning_content="think"), delta(content="ok")), False)])
    turn = p.stream(model="m", messages=MSG)
    assert turn.reasoning == "think" and turn.reasoning_field == "reasoning_content" and turn.text == "ok"


def test_tool_call_arrives_in_fragments(serve):
    frag = lambda **fn: delta(tool_calls=[{"index": 0, **fn}])
    _s, p = serve([(200, sse(
        frag(id="call_1", type="function", function={"name": "read_file", "arguments": ""}),
        frag(function={"arguments": '{"path": "a'}),
        frag(function={"arguments": '.txt"}'}),
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}), False)])
    turn = p.stream(model="m", messages=MSG)
    assert [(c.id, c.name, c.arguments) for c in turn.tool_calls] == [("call_1", "read_file", {"path": "a.txt"})]
    assert turn.finish_reason == "tool_calls"


def test_two_calls_with_the_same_index_but_different_ids(serve):
    one = lambda i, name: delta(tool_calls=[{"index": 0, "id": i, "function": {"name": name, "arguments": "{}"}}])
    _s, p = serve([(200, sse(one("a", "list_files"), one("b", "grep")), False)])
    turn = p.stream(model="m", messages=MSG)
    assert [(c.id, c.name) for c in turn.tool_calls] == [("a", "list_files"), ("b", "grep")]


def test_broken_tool_arguments_are_kept_raw_not_raised(serve):
    _s, p = serve([(200, sse(delta(tool_calls=[{"index": 0, "id": "x", "function": {"name": "t", "arguments": '{"a": '}}])), False)])
    turn = p.stream(model="m", messages=MSG)
    assert turn.tool_calls[0].arguments == {"_raw": '{"a": '}


def test_error_inside_a_200_stream(serve):
    _s, p = serve([(200, sse({"error": {"message": "boom"}}), False)])
    with pytest.raises(ProviderError, match="boom"):
        p.stream(model="m", messages=MSG)


def test_stream_that_ends_without_done_is_an_error(serve):
    _s, p = serve([(200, sse(delta(content="half"), done=False), False)])
    with pytest.raises(ProviderError, match="without"):
        p.stream(model="m", messages=MSG)


def test_server_that_rejects_stream_options_gets_a_second_request_without_them(serve):
    server, p = serve([(400, b'{"error": "unknown field stream_options"}', False),
                       (200, sse(delta(content="ok")), False)])
    assert p.stream(model="m", messages=MSG).text == "ok"
    assert "stream_options" in server.requests[0] and "stream_options" not in server.requests[1]


def test_rate_limit_is_retried(serve):
    server, p = serve([(429, b'{"error": "slow down"}', False), (200, sse(delta(content="ok")), False)])
    assert p.stream(model="m", messages=MSG).text == "ok"
    assert len(server.requests) == 2


def test_a_plain_400_is_not_retried(serve):
    server, p = serve([(400, b'{"error": "bad request"}', False), (200, sse(delta(content="ok")), False)])
    with pytest.raises(ProviderError) as info:
        p.stream(model="m", messages=MSG)
    assert info.value.status == 400 and len(server.requests) == 1


def test_cancel_while_the_server_is_silent_returns_quickly_with_the_partial_text(serve):
    _s, p = serve([(200, sse(delta(content="so far"), done=False), True)])
    cancel = threading.Event()
    threading.Timer(0.4, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(StreamCancelled) as info:
        p.stream(model="m", messages=MSG, cancel=cancel)
    assert time.monotonic() - started < 3
    assert info.value.partial.text == "so far"


def test_complete_without_streaming(serve):
    body = json.dumps({"choices": [{"message": {"content": "plain"}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 5, "completion_tokens": 2}}).encode()
    server, p = serve([(200, body, False)])
    turn = p.complete(model="m", messages=MSG)
    assert turn.text == "plain" and "stream" not in server.requests[0]


def test_a_rejected_default_ceiling_is_dropped_and_not_offered_again(serve):
    server, p = serve([(400, b'{"error": "Invalid max_tokens value, the valid range of max_tokens is [1, 8192]"}', False),
                       (200, sse(delta(content="one")), False), (200, sse(delta(content="two")), False)])
    assert p.stream(model="m", messages=MSG).text == "one"
    assert p.stream(model="m", messages=MSG).text == "two"
    assert "max_tokens" in server.requests[0] and "max_tokens" not in server.requests[1]
    assert "max_tokens" not in server.requests[2]  # learned: the second call did not even try


def test_a_ceiling_the_user_set_is_never_silently_dropped(serve):
    server, p = serve([(400, b'{"error": "max_tokens too large"}', False)])
    with pytest.raises(ProviderError):
        p.stream(model="m", messages=MSG, max_tokens=999999)
    assert len(server.requests) == 1


def test_default_ceiling_is_the_same_for_every_backend_until_one_is_known(monkeypatch):
    from minicw.core.provider import OpenAICompatProvider as P
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    assert P.for_backend("deepseek").default_max_tokens == P.for_backend("ollama").default_max_tokens == 32000
