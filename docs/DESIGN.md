# minicw: how it works

One file, read top to bottom. For each part: what it does, which OpenWorker file it follows
(`reference/openworker/coworker/...` @ 59151e3, MIT), and what is simpler or different here and why.
"mcw" below is the earlier one-shot harness in this repository (`mcw/`); several modules here began as
copies of it. Line numbers are those of the reference at that commit.

## 0. The shape of a turn

```
you type a line
  └─ Repl            ui/repl.py        reads the line, calls engine.run(line)
       └─ Engine.run core/engine.py    one turn:
            1. repair()                close tool calls left without a result by an old crash
            2. append the user message  (and write it to the session file at once)
            3. loop:
                 compaction due?  → summarise old messages (outbound view only)
                 provider.stream() → text arrives as deltas, tool calls as fragments
                 append the assistant message
                 no tool calls?  → done
                 for each tool call:
                     permissions.evaluate()  → allow / ask / deny
                     ask → approver (a prompt on the terminal)
                     run the tool in a worker thread, watching the cancel flag
                     append the tool result (cut / spilled to a file if huge)
```

Ctrl-C sets a flag; every step above looks at it. State that outlives the process is the session file
(messages, one per line) plus a small metadata file (mode, grants, compaction, token use).

## 1. core/

### provider.py: one client for every backend, with streaming
Follows `providers/openai_provider.py` (the default output cap, dropping a parameter the server rejects,
reasoning replay) and, for streaming, `:346-451` (accumulating tool-call fragments by `index`).
Deepseek, Ollama and Groq all speak OpenAI's Chat Completions, so one class covers them.
* **Streaming in a worker thread.** Python cannot interrupt a blocking socket read on Windows. The worker
  reads the server-sent events and puts chunks on a queue; the caller polls the queue every 0.1 s and can
  look at the cancel flag between polls. On cancel the worker is abandoned and closes its own response.
* `StreamAccumulator` joins fragments: text and thinking become deltas at once; a tool call arrives as id and
  name, then JSON argument fragments; fragments are joined per `index`, and a new non-empty id at an index
  that already has one starts a new call (Ollama sends every call with index 0).
* Retries: 429, 5xx, network errors, and a 400 that is really the endpoint rejecting the model's malformed
  tool call, after 1, 3, 8 seconds (`Retry-After` respected). Only before any text has been shown; a failure
  half-way through a reply is an error, not a silent restart that would print the text twice.
* A 400 that names `stream_options`, `reasoning_effort` or our default `max_tokens` gets one more request
  without that parameter (`:129-154`).
* Tool-call arguments that are not valid JSON are kept under the key `_raw` and the call is not run; the
  model is told what went wrong.
Different from the reference: no SDK (aisuite/OpenAI), only `urllib`; synchronous.

### engine.py: the loop
Follows `engine.py` (`run` `:382`, `request_interrupt` `:332`, `_repair_dangling_tool_calls` `:576`) in
behaviour, but is about 600 lines instead of 3,000: no reviewer model, steering, parallel tools, teams,
sandbox, durable approvals.
Invariants it keeps (each exists because a weak model breaks it): every tool call gets exactly one result
with the matching id; unparsable arguments are never executed; a reply cut off at the output limit is
continued; text that looks like a tool call but did not parse ends the turn as an error; compaction changes
only the copy that is sent, never `self.messages`.
* **Interruption.** `run()` installs a SIGINT handler for the turn (main thread only) that sets
  `cancel`; a second Ctrl-C re-raises KeyboardInterrupt, so a stuck program can still be killed.
  Pending calls are closed with an error result the moment the interrupt is seen, so an orphan can only be
  at the tail of the history and `repair()` normally only appends. Blocking work (tools, the compaction
  summary) runs in a thread; the engine waits in 0.1 s slices and, after cancel, gives the work 2 seconds
  to finish by itself (the shell tool kills its process tree) before abandoning it.
* **Plan mode** is a mode of the permission engine plus a notice added to the OUTBOUND copy of the last user
  message (never stored, never in the system prompt, which must stay byte-identical for the prompt cache).
  `propose_plan` and `ask_user` are answered by the engine on the main thread, because they talk to a person.
Different from mcw: events go through a callback (`on_event`) as in mcw, not a generator; no workflow,
verifier, lessons, spend ledger.

### permissions.py, risk.py, readonly.py: what may run
`risk.py`: a tool must declare one of READ, EGRESS, WRITE_LOCAL, EXEC, EXTERNAL (reference `risk.py:18-28`).
There is no default: a tool nobody classified would otherwise pass as harmless.
`permissions.py` is a table (risk class × mode, see the README). The mode is checked first and grants second,
so an "always" given in ask mode cannot let a write through in plan mode. Floors that no mode lowers
(`permissions.py:150-166` in the reference): destructive shell patterns, writes outside the workspace,
files that run later. `readonly.py` (copied from mcw, itself from the reference's `readonly.py`) decides
whether a shell command only reads: one simple command from a short allowlist, no pipes or redirects, no
flag that writes, and every file it names must be inside the workspace.
Known gap: the allowlist contains `grep`, `find`, which do not exist in PowerShell; the command is allowed
without asking and then fails harmlessly.

### sessions.py: a conversation that outlives the process
Follows `conversations.py` and `sessions.py`: one `.jsonl` per conversation, append-only, and a metadata
record. Simpler: a JSON file instead of a SQLite index. Differences that matter:
* each message is written when it appears (the reference appends the new messages at the end of the turn);
* the metadata is written AFTER the messages it describes;
* a torn last line (the process died mid-write) is dropped and the file repaired atomically on open;
* a saved compaction state whose `boundary` does not fit the history is ignored;
* the id becomes a file name, so `[A-Za-z0-9_-]{1,128}` only (reference `conversations.py:30-34`).
Everything written goes through `redact.py` (API keys, `sk-...` tokens, secret-looking environment values).

### compaction.py: summarising old messages
Copied from mcw (`compaction.py`), which follows the reference's `compaction.py`. When the prompt nears 80%
of the context window the older part is replaced, in what is SENT, by an LLM-written summary plus facts
extracted by code (files written/read, commands run, the user's own messages). The first user message is
always kept; cuts fall before any non-tool message so a call is never separated from its result. The state
(`as_dict`/`from_dict`, reference `:164-191`) is saved with the session. A summary that is too short, missing
sections, or cut off is retried once; if no summary can be made, only about 10% of the messages are dropped.
The summary is re-summarised at each compaction; detail from the oldest turns fades. This is a known property
of the approach, not a bug.

### prompt.py: the system prompt, built once
Base instructions (with the "one sentence before a batch of tool calls" rule, reference `_NARRATION_GUIDANCE`),
the persona, the environment (folder, OS, shell, date, a git snapshot taken once and labelled as such),
`AGENTS.md` (global then project), `rules.md`, memory, the skills catalog. Everything here is stable during a
conversation so the provider's prompt cache keeps working (reference `agent.py:516-563`).

## 2. tools/

| Module | Tools | Notes |
|---|---|---|
| files.py | read_file, list_files, grep, write_file, replace_in_file | from mcw, which copied the reference; reads are windowed, results are dicts with an `error` key on failure |
| shell.py | run_shell | PowerShell on Windows, bash elsewhere; secret-looking environment variables are not passed on; the process tree is killed on cancel or timeout |
| todo.py | todo_write | the argument is `todos`, never `items` (a key named `items` breaks some chat templates, reference `todo.py:16-20`) |
| system.py | git_status, git_diff, git_log, current_time, runtime_context | time is a tool, not prompt text, to keep the prompt cache valid |
| web.py | web_search, web_fetch | guard below |
| interact.py | ask_user, propose_plan | registered for their schemas; the engine answers them |
| memory.py | remember, memory_read/update/forget/search | SQLite, FTS5 with a LIKE fallback; global and workspace scope |
| skills.py | load_skill, save_skill | catalog in the prompt, bodies on demand |
| subagent.py | explore | a read-only helper with its own context |
| results.py | (not a tool) | a result longer than 10,000 characters is written to `<state>/spill/` and the model gets the ends plus the path |

* **Web guard** (reference `web/guard.py:40-178`): only http/https; a host whose DNS answers include a private,
  loopback, link-local (169.254.169.254 too), CGNAT, multicast or reserved address is refused; redirects are
  followed by hand, at most 5, each hop checked again. Not done: pinning the connection to the checked IP, which
  stops DNS rebinding in the reference (it needs httpx).
* **Memory** (reference `memory/`, `agent.py:76-99`): no `session` scope (it was never read back). The block in the
  prompt is the full text while short, an index of one-line summaries past 8,000 characters (the index itself has
  no length cap, as in the reference). Tools that write ask even in auto mode.
* **explore** (reference `tools/subagent.py:79-138`): shares the parent's cancel event, so one Ctrl-C stops both.
  It runs inside the parent's tool thread, so it cannot ask the user; its approver says no. Web access in a helper
  therefore needs auto mode or an "always" given in the main conversation.
* **Spill** (reference `toolresult.py`, OPE-186): the spill folder is a read-only root of the workspace, so
  `read_file` can open it and nothing can write there. `read_file` results are cut in the middle instead of
  spilled (a spilled spill would loop).

## 3. integrations/

### mcp.py: tools from other programs
Reference: `mcp/client.py:208-223` (flattening results), `mcp/tools.py:26-86` (naming, schema pass-through,
the remote name held in a closure so a model cannot reroute a call). The reference uses the official async SDK
and a thread-to-event-loop bridge; this is a hand-written synchronous stdio client, about 250 lines, so the
whole protocol is readable: JSON-RPC lines, `initialize` → `notifications/initialized` → `tools/list` →
`tools/call`. Servers that print junk on stdout are tolerated; requests from the server are answered "method not
supported"; a reply that arrives after its call timed out is dropped; a dead server is detected. Windows:
`npx` is `npx.cmd`; the child gets its own process group; the tree is killed with `taskkill /T`.
Not supported: HTTP/SSE servers, OAuth, resources, prompts.

### google/: Gmail and Calendar
Reference: `connectors/integration_tools.py:1099-1509`. The same tools (search, read, send; list, create,
update, delete), called straight through Google's API. The reference signs users in through a cloud broker; here
you use your own OAuth client (steps in the README) and `InstalledAppFlow.run_local_server(port=0)`. The
Google libraries are optional and imported only in `auth.py`; the tool code takes an injected `get_service`, so
tests use fakes. The consent-screen "Testing" mode expires the refresh token after 7 days.

## 4. ui/

`repl.py` is a loop around `engine.run`; `render.py` turns events into text; `commands.py` holds the slash
commands. Rules: nothing is printed while a prompt is open; every question during a turn is a prompt_toolkit
prompt on the main thread (never `input()`), so Ctrl-C there means "no, and stop the turn"; model text is
printed with markup off (a reply containing `[/x]` must not crash rich); no spinner and no `rich.Live`. When
prompt_toolkit cannot use the terminal (Git Bash on Windows, piped input) the loop falls back to `input()`.
The reference's TUI is Textual (`tui/app.py`) and async; this one is synchronous on purpose.

## 5. What is deliberately missing

GUI, local server, sandbox, agent teams, remote machines, Slack and the other connectors, scheduled tasks,
browser automation, the reviewer model for auto-approval, mid-turn steering, `apply_patch`, attachments,
trajectory export, cost tracking. Each of these is infrastructure or product around the loop, not the loop.
