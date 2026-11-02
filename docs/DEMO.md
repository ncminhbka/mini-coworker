# Demo script

About 10 minutes. Do the "before" section the day before and again an hour before.

## Before

1. `minicw doctor` shows ✓ for the key and the model. The model line sends one tiny request.
2. `DEEPSEEK_API_KEY` is in `.env` (never on screen). Do not run `echo` of it in the demo.
3. Google: `/google login` the day before (the sign-in expires after 7 days in Testing mode). `/google` says
   "signed in".
4. MCP: `/mcp` shows the server as `ready`. The first `npx -y` downloads the package; run it once beforehand.
5. A demo folder with a small project (a few Python files and a failing test), already `git init`ed.
6. Fallback if the network fails: `/backend ollama mcw-qwen3.5-4b` (Ollama running, model pulled).
7. Close other programs that print into the terminal. Use a wide window (110+ columns).

## Script

| Step | Do | What it shows |
|---|---|---|
| 1 | `cd demo` then `minicw --persona coder` | banner: model, mode, folder, session id; the persona |
| 2 | "What does this project do? Give me the key files." | streaming; narration sentence before tools; `list_files`, `read_file` run without asking (read-only) |
| 3 | "Run the tests and tell me what fails." | `run_shell` asks: answer `y`. The permission panel, the risk, the reason |
| 4 | "Fix it." | `replace_in_file` asks: answer `a` (always for this tool); later edits do not ask |
| 5 | `/mode plan`, "Add a CLI flag --verbose" | plan mode: it explores, cannot write, ends with `propose_plan`; approve with `y` and it implements |
| 6 | "Try to delete everything: rm -rf ." | refused by the floor, whatever the mode |
| 7 | "Search the web for what the MCP protocol is and summarise with sources." | `web_search`, `web_fetch`, answer with URLs |
| 8 | `/persona assistant`, "What is on my calendar this week?" | Google tools; reading asks once (answer `a`) |
| 9 | "Create a 30 minute event tomorrow 9:00 called demo prep." | EXTERNAL: always asks; show the details before `y` |
| 10 | "Remember that I prefer short answers." | `remember` asks (even in auto); `/memory` lists it |
| 11 | `Ctrl-D`, then `minicw --continue`, ask "what were we doing?" | the session was saved after every message and reopened |
| 12 | `/sessions`, `/tokens`, then open `~/.minicw/audit.jsonl` | the audit trail: who allowed what, and why |

## Ctrl-C rehearsal (do each once before the demo)

* While text is streaming: the turn stops, the text so far stays, the prompt returns.
* While a shell command runs (`Start-Sleep 30`): it stops within a second or two; no `powershell` or `python`
  process stays behind (check Task Manager once).
* At a permission question: Ctrl-C means "no" and ends the turn.
* At the prompt: Ctrl-C clears the line and does nothing else; Ctrl-D leaves.
* Stuck? A second Ctrl-C in the same turn forces KeyboardInterrupt.

## If something goes wrong on stage

* Rate limit or network error: the error line shows the reason; the session is intact; try again or
  `/backend ollama`.
* A tool result looks wrong: `minicw --debug` logs the redacted requests and replies to `<state>/debug.jsonl`.
* Never paste the `.env` file or a debug log without checking it, even though keys are redacted.

## What to say about the code

The loop is `core/engine.py`; the safety table is `core/permissions.py`; the streaming client is
`core/provider.py`; `docs/DESIGN.md` says, for each part, which OpenWorker file it follows and what is
different. The tests (`pytest`) include real Ctrl-C signals and a fake MCP server.
