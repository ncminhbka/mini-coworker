"""A small MCP (Model Context Protocol) client for servers started as local programs (stdio).

MCP lets the assistant use tools that live in another program: you list servers in
`<state>/mcp.json`, in the same shape Claude Desktop uses,

    {"mcpServers": {"files": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]}}}

and minicw starts each one, asks it for its tools (`tools/list`) and offers them to the model as
`mcp__<server>__<tool>`. A call is `tools/call`. The wire format is JSON-RPC 2.0, one JSON
message per line on the server's stdin/stdout. Handshake: `initialize` -> reply ->
`notifications/initialized`.

The reference uses the official async `mcp` SDK (`mcp/client.py`, `mcp/tools.py` @ 59151e3,
MIT) and a thread-to-event-loop bridge. This is hand-written and synchronous instead (about 250
lines), which is the point of a study version: you can read the whole protocol here. Not
supported: HTTP/SSE servers, OAuth, paging beyond `nextCursor`, resources and prompts.
What is kept from the reference: the naming scheme and 64-character limit, descriptions cut at
1024, schemas passed through unchanged, text blocks joined, `isError` becoming an error result,
and the remote tool name held in a closure (never an argument the model could change).

Windows details that matter: `npx` is really `npx.cmd` (found with shutil.which); the child starts
in its own process group so Ctrl-C does not make cmd.exe ask "Terminate batch job (Y/N)?" on our
pipe; the whole tree is killed with taskkill; stdout/stdin are binary pipes with explicit UTF-8;
stderr goes to a log file so a full pipe can never block the server.

Every MCP tool is Risk.EXTERNAL: a server can do anything, so each call asks the user unless they
answered "always" for that tool.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..core.risk import Risk
from ..tools.registry import ToolSpec
from ..tools.shell import kill_tree

PROTOCOL_VERSION = "2024-11-05"
START_TIMEOUT = 60.0  # `npx -y` may have to download the server the first time
CALL_TIMEOUT = 120.0
MAX_NAME = 64
MAX_DESCRIPTION = 1024


class McpError(Exception):
    pass


@dataclass
class ServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    include_tools: Optional[list[str]] = None
    exclude_tools: list[str] = field(default_factory=list)


_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: str) -> str:
    """${VAR} -> the environment variable (an unset one becomes empty)."""
    return _VAR.sub(lambda m: os.environ.get(m.group(1), ""), value)


def load_config(path: Path) -> tuple[list[ServerConfig], list[str]]:
    """(servers, problems). A missing file is no servers; a bad entry is reported and skipped."""
    if not path.is_file():
        return [], []
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))  # utf-8-sig: Windows PowerShell's Set-Content adds a BOM
    except (OSError, ValueError) as exc:
        return [], [f"{path.name}: {exc}"]
    servers, problems = [], []
    for name, entry in (data.get("mcpServers") or {}).items():
        if not isinstance(entry, dict) or entry.get("enabled") is False:
            continue
        if entry.get("url") or entry.get("type") in ("http", "sse"):
            problems.append(f"{name}: HTTP/SSE servers are not supported (only a local command)")
            continue
        if not entry.get("command"):
            problems.append(f"{name}: no 'command'")
            continue
        servers.append(ServerConfig(
            name=str(name), command=_expand(str(entry["command"])), args=[_expand(str(a)) for a in entry.get("args", [])],
            env={str(k): _expand(str(v)) for k, v in (entry.get("env") or {}).items()},
            cwd=entry.get("cwd"), include_tools=entry.get("include_tools"),
            exclude_tools=list(entry.get("exclude_tools") or [])))
    return servers, problems


class StdioClient:
    """One running server. A reader thread turns the server's stdout lines into replies for
    waiting requests; everything else on stdout is ignored."""

    def __init__(self, config: ServerConfig, log_path: Path) -> None:
        self.config = config
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen] = None
        self.tools: list[dict[str, Any]] = []
        self.error = ""
        self._next_id = 0
        self._lock = threading.Lock()
        self._waiting: dict[int, threading.Event] = {}
        self._replies: dict[int, dict[str, Any]] = {}
        self._log = None

    # -- process ---------------------------------------------------------------------------

    def start(self) -> None:
        exe = shutil.which(self.config.command)
        if exe is None:
            raise McpError(f"command not found: {self.config.command}")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "ab")
        env = {**os.environ, **self.config.env}
        self.proc = subprocess.Popen(
            [exe, *self.config.args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
            cwd=self.config.cwd or None, env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
            start_new_session=sys.platform != "win32")
        threading.Thread(target=self._read_loop, daemon=True, name=f"mcp-{self.config.name}").start()

    def _read_loop(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for raw in iter(self.proc.stdout.readline, b""):
            try:
                message = json.loads(raw.decode("utf-8", errors="replace"))
            except ValueError:
                continue  # a banner or log line on stdout: not protocol
            if not isinstance(message, dict):
                continue
            if "method" in message:
                if "id" in message:  # the server asks us something (roots/list, ping...): we support none
                    self._send({"jsonrpc": "2.0", "id": message["id"],
                                "error": {"code": -32601, "message": "method not supported by this client"}})
                continue  # notifications are ignored
            mid = message.get("id")
            with self._lock:
                event = self._waiting.pop(mid, None)
                if event is not None:
                    self._replies[mid] = message
            if event is not None:
                event.set()  # a reply nobody waits for any more (a timed-out call) is dropped
        try:  # stdout closed: the process is ending. Let the OS finish reporting it, so alive() is right
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        with self._lock:  # wake everybody who is still waiting for an answer
            for event in self._waiting.values():
                event.set()

    def _send(self, message: dict[str, Any]) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        try:
            self.proc.stdin.write((json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8"))
            self.proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise McpError(f"{self.config.name}: server is not running ({exc})") from exc

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def close(self) -> None:
        if self.proc is not None:
            kill_tree(self.proc)  # the whole tree: cmd.exe -> node -> node
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
        if self._log is not None:
            self._log.close()

    # -- protocol --------------------------------------------------------------------------

    def request(self, method: str, params: Optional[dict[str, Any]] = None, timeout: float = CALL_TIMEOUT) -> Any:
        with self._lock:
            self._next_id += 1
            mid = self._next_id
            event = self._waiting[mid] = threading.Event()
        self._send({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}})
        if not event.wait(timeout):
            with self._lock:
                self._waiting.pop(mid, None)
            raise McpError(f"{self.config.name}: no answer to {method} after {timeout:.0f}s")
        with self._lock:
            reply = self._replies.pop(mid, None)
        if reply is None:
            raise McpError(f"{self.config.name}: the server stopped (see {self.log_path})")
        if "error" in reply:
            raise McpError(f"{self.config.name}: {reply['error'].get('message', reply['error'])}")
        return reply.get("result") or {}

    def initialize(self, timeout: float = START_TIMEOUT) -> None:
        self.request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                                    "clientInfo": {"name": "minicw", "version": "0.1.0"}}, timeout)
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})  # required, easy to forget

    def list_tools(self, timeout: float = 20.0) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor = None
        while True:
            result = self.request("tools/list", {"cursor": cursor} if cursor else {}, timeout)
            tools += result.get("tools") or []
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    def call_tool(self, name: str, arguments: dict[str, Any], timeout: float = CALL_TIMEOUT) -> dict[str, Any]:
        result = self.request("tools/call", {"name": name, "arguments": arguments}, timeout)
        parts = []
        for block in result.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, dict):
                parts.append(f"[{block.get('type', 'content')}]")  # images and the like: a placeholder
        body = "\n".join(parts)
        if not body and result.get("structuredContent") is not None:
            body = json.dumps(result["structuredContent"], ensure_ascii=False)
        return {"error": body or "tool failed"} if result.get("isError") else {"result": body}


def tool_name(server: str, tool: str, taken: set[str]) -> str:
    """mcp__<server>__<tool>: characters outside [A-Za-z0-9_-] become "_", 64 characters at most
    (OpenAI's limit); a clash gets a numeric suffix."""
    base = re.sub(r"[^A-Za-z0-9_-]", "_", f"mcp__{server}__{tool}")[:MAX_NAME]
    name, n = base, 1
    while name in taken:
        n += 1
        suffix = f"_{n}"
        name = base[: MAX_NAME - len(suffix)] + suffix
    taken.add(name)
    return name


def _invoker(client: StdioClient, remote: str):
    """The function behind one MCP tool. The remote tool name lives in this closure, NOT in a
    parameter: if it were a parameter (even one with a default), a model could pass a different
    value and reach another tool of the server."""
    def invoke(**arguments: Any) -> dict[str, Any]:
        return client.call_tool(remote, arguments)

    return invoke


def make_tool_specs(client: StdioClient, taken: set[str]) -> list[ToolSpec]:
    cfg = client.config
    specs = []
    for tool in client.tools:
        remote = str(tool.get("name", ""))
        if not remote or (cfg.include_tools is not None and remote not in cfg.include_tools) \
                or remote in cfg.exclude_tools:
            continue
        schema = tool.get("inputSchema") if isinstance(tool.get("inputSchema"), dict) else {}
        schema = {"type": "object", "properties": {}, **schema} if schema.get("type") in (None, "object") else schema

        specs.append(ToolSpec(
            tool_name(cfg.name, remote, taken),
            f"[MCP server {cfg.name}] " + str(tool.get("description") or remote)[:MAX_DESCRIPTION],
            schema, _invoker(client, remote), Risk.EXTERNAL))
    return specs


@dataclass
class ServerStatus:
    name: str
    state: str  # starting | ready | failed
    detail: str = ""
    tools: int = 0


class McpManager:
    """Starts the configured servers in parallel and collects their tools."""

    def __init__(self, state: Path) -> None:
        self.state = state
        self.clients: dict[str, StdioClient] = {}
        self.status: dict[str, ServerStatus] = {}
        self.problems: list[str] = []
        self._threads: list[threading.Thread] = []
        atexit.register(self.close)

    def start(self, wait: float = 30.0) -> None:
        """Start every server and wait up to `wait` seconds for them. Slow ones keep starting in
        the background; their tools appear the next time an engine is built."""
        configs, self.problems = load_config(self.state / "mcp.json")
        for cfg in configs:
            self.status[cfg.name] = ServerStatus(cfg.name, "starting")
            thread = threading.Thread(target=self._boot, args=(cfg,), daemon=True, name=f"mcp-boot-{cfg.name}")
            thread.start()
            self._threads.append(thread)
        deadline = time.monotonic() + wait
        for thread in self._threads:
            thread.join(max(0.0, deadline - time.monotonic()))

    def _boot(self, cfg: ServerConfig) -> None:
        client = StdioClient(cfg, self.state / "logs" / f"mcp-{cfg.name}.log")
        try:
            client.start()
            client.initialize()
            client.tools = client.list_tools()
        except McpError as exc:
            client.close()
            self.status[cfg.name] = ServerStatus(cfg.name, "failed", str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - one bad server must not stop the others
            client.close()
            self.status[cfg.name] = ServerStatus(cfg.name, "failed", f"{type(exc).__name__}: {exc}")
            return
        self.clients[cfg.name] = client
        self.status[cfg.name] = ServerStatus(cfg.name, "ready", tools=len(client.tools))

    def tool_specs(self) -> list[ToolSpec]:
        taken: set[str] = set()
        specs: list[ToolSpec] = []
        for client in self.clients.values():
            if client.alive():
                specs += make_tool_specs(client, taken)
        return specs

    def close(self) -> None:
        for client in self.clients.values():
            client.close()
        self.clients.clear()
