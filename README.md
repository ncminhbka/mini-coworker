# minicw

A small terminal AI coworker. It reads and writes files in a folder, runs commands, searches the
web, remembers things, can use MCP servers and your Gmail and Calendar, and always asks before it does
anything that matters. It is a study reimplementation of the core of
[OpenWorker](https://github.com/andrewyng/openworker) (MIT), written to be read: a few thousand lines of
Python (comments included), no agent framework, and a design note for every part in [docs/DESIGN.md](docs/DESIGN.md).

```
minicw · deepseek/deepseek-flash · ask · D:\work\demo
session 3f2a9c1b7d04   /help for commands, Ctrl-D to leave
[deepseek-flash · ask] › summarise the TODOs in this repo
● I'll search the code for TODO markers first.
→ grep {"pattern": "TODO", "glob": "*.py"}
  ✓ grep · ok  {"matches": ["app.py:12:# TODO ..."], ...
● There are 3 TODOs: ...
```

## Install

Python 3.11 or newer.

```
cd minicw
pip install -e ".[dev]"          # the program, rich, prompt_toolkit, ddgs, pytest
pip install -e ".[google]"       # optional: Gmail and Calendar
```

## Configure

Keys go in a `.env` file in the folder you run from, or in `~/.minicw/.env` (a real environment
variable wins over both). Never commit it.

```
DEEPSEEK_API_KEY=...            # for the default backend
MINICW_STATE_DIR=D:/minicw-state   # optional: where sessions, memory, logs live (default ~/.minicw)
```

Backends: `deepseek` (needs the key), `ollama` (local, `http://localhost:11434`, no key), `groq`.
Check everything with:

```
minicw doctor
```

## Use

```
minicw                         # new session in the current folder
minicw --continue              # reopen the newest session of this folder
minicw --resume <id>           # reopen a given one
minicw --persona coder         # play a role (see Personas)
minicw --backend ollama --model mcw-qwen3.5-4b --context-window 16384
minicw run --prompt "..."      # one task, no questions, the answer on stdout (exit 0 = completed)
```

Inside: type a request. `/help` lists the commands:

| Command | What it does |
|---|---|
| `/mode plan\|ask\|auto\|bypass` | how much the assistant may do without asking (below) |
| `/model <name>`, `/backend <name> [model]` | change model or provider (`/backend ollama` is the offline fallback) |
| `/new`, `/sessions`, `/resume <id>` | conversations; they are saved after every message |
| `/persona [id\|none]` | list or switch roles |
| `/memory`, `/skills` | what is remembered; skills available |
| `/mcp [reload]`, `/google [login]` | MCP servers and Google sign-in |
| `/tokens` | token use |
| `Ctrl-C` | stops the current turn (not the program); press twice to force |
| `Ctrl-D`, `/exit` | leave |

## Modes and permissions

Every tool declares a risk class; the mode decides what that class may do without asking.

| Mode | read | web | write files | run commands | send mail, change calendar, MCP |
|---|---|---|---|---|---|
| `plan` | yes | yes | no | read-only only | no |
| `ask` (default) | yes | asks | asks | read-only yes, others ask | asks |
| `auto` | yes | yes | yes, in the workspace | asks | asks |
| `bypass` | yes | yes | yes | yes | yes |

At a question: `y` once, `a` always (this tool, or this exact command), anything else is no.
In every mode: destructive commands (`rm -rf`, `git push --force`, `curl | sh`...) are refused, writes
outside the workspace are refused, and files that run later (`.git/hooks`, CI files) always ask except
in `bypass`. In `plan` mode the assistant explores, then calls `propose_plan`; if you approve, the mode
becomes `ask` and it carries the plan out. Every decision is appended to `<state>/audit.jsonl`.

## Personas, skills, memory

* **Persona**: a folder with `manifest.md` (`id`, `name`, optional `tools: [...]`, `skills: [...]`, and
  the body as the role's prompt). Built in: `coder`, `assistant`. Yours go in `<state>/personas/<id>/`.
* **Skill**: a folder with `SKILL.md` (frontmatter `name`, `description`; the body is the steps). Only the
  catalog is in the prompt; the assistant opens one with `load_skill`. Put them in `<state>/skills/`,
  `<workspace>/.minicw/skills/`, or inside a persona. The assistant can propose one with `save_skill`.
* **Memory**: `remember` saves a fact (about you, or about this project); it is shown to the assistant in
  every later session. Saving asks you. `/memory forget <id>` deletes.
* `AGENTS.md` in the project and `<state>/AGENTS.md` are added to the prompt; `<state>/rules.md` holds
  your standing rules.

## MCP servers

`<state>/mcp.json`, the same shape as Claude Desktop's:

```json
{"mcpServers": {
  "files": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "D:/work"]}
}}
```

Local programs only (stdio). Their tools appear as `mcp__files__read_file` and so on, and each call asks
you. `/mcp` shows which servers started.

## Gmail and Calendar

1. Google Cloud console: create a project, enable the **Gmail API** and the **Google Calendar API**.
2. OAuth consent screen: type External, status Testing, add yourself as a test user.
3. Credentials: create an OAuth client ID of type **Desktop app**, download the JSON.
4. Save it as `<state>/google/credentials.json`, run `pip install -e ".[google]"`, then `/google login`.

While the consent screen is in Testing, Google expires the sign-in after 7 days: log in again before a demo.
Reading mail and the calendar asks like any web access; sending, creating, changing and deleting always ask.

## Tests

```
pytest            # about 130 tests, offline, no API key needed
```

They include real Ctrl-C (a signal sent to the process while it streams, while a shell command runs, and
while it waits at a permission question), a local server that speaks the streaming protocol, and a fake MCP
server.

## Where things are

See [docs/DESIGN.md](docs/DESIGN.md) for each module: what it does, which OpenWorker file it follows, and
what is simpler or different here. [docs/DEMO.md](docs/DEMO.md) is a script for a live demo.
