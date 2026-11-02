"""A tiny MCP server over stdio, used by the tests. Speaks JSON-RPC, one message per line.

Behaviours on purpose: prints a junk line on stdout first; asks the client `roots/list` after the
handshake (and logs the client's reply to $FAKE_MCP_LOG); tools: echo, boom (isError), slow
(sleeps), die (exits in the middle of a call).
"""

import json
import os
import sys
import time

LOG = os.environ.get("FAKE_MCP_LOG")


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def log(text):
    if LOG:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


TOOLS = [
    {"name": "echo", "description": "Echo the text back.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "boom", "description": "Always fails.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "slow", "description": "Sleeps.", "inputSchema": {"type": "object", "properties": {"seconds": {"type": "number"}}}},
    {"name": "die", "description": "Kills the server.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "bad name!", "description": "Name needs cleaning.", "inputSchema": {"type": "object", "properties": {}}},
]

print("booting fake server, this line is not JSON-RPC", flush=True)
for line in sys.stdin:
    try:
        msg = json.loads(line)
    except ValueError:
        continue
    method, mid = msg.get("method"), msg.get("id")
    if method is None:  # a reply to something we asked
        log(f"client replied to {mid}: {json.dumps(msg.get('error') or msg.get('result'))}")
        continue
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                                                       "serverInfo": {"name": "fake", "version": "1"}}})
    elif method == "notifications/initialized":
        send({"jsonrpc": "2.0", "id": "srv1", "method": "roots/list"})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
    elif method == "tools/call":
        name, args = msg["params"]["name"], msg["params"].get("arguments") or {}
        log(f"call {name} {json.dumps(args, sort_keys=True)}")
        if name == "echo":
            send({"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": str(args.get("text"))},
                                                                         {"type": "image", "data": "x", "mimeType": "image/png"}]}})
        elif name == "boom":
            send({"jsonrpc": "2.0", "id": mid, "result": {"isError": True, "content": [{"type": "text", "text": "kaboom"}]}})
        elif name == "slow":
            time.sleep(float(args.get("seconds", 1)))
            send({"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": "late"}]}})
        elif name == "die":
            os._exit(3)
        else:
            send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"unknown tool {name}"}})
    elif mid is not None:
        send({"jsonrpc": "2.0", "id": mid, "result": {}})
