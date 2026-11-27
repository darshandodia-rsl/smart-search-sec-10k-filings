---
name: extract-run
description: Start one or more LLM extraction backends (groq, agy, sonnet) on the smart-search-sec-10k-filings project. Each backend gets its own background runner under caffeinate in stop-when-blocked mode and works through the queued companies; watches events to send phone pushes. Companies are added separately with /extract-add. Use only when the user invokes /extract-run.
disable-model-invocation: true
argument-hint: "groq | agy | sonnet [PCT] [--over-weekly]   (one or more, e.g. `agy sonnet 60`)"
---

# Extract run

Spends LLM quota. Only ever run because the user typed /extract-run. Run from the project root;
say in one line what you are about to do before each step that changes anything.

Starts **exactly the backends the user named**, each as its **own runner process**:

| Name     | Backend           | Notes |
|----------|-------------------|-------|
| `groq`   | `groq`            | |
| `agy`    | `antigravity_cli` | |
| `sonnet` | `claude_cli`      | optional `PCT` = 5-hour allowance (1–100), `--over-weekly` only if the user wrote it |

No names given: ask which backends to start (do not pick for the user). Unknown name: say so and stop.

Every runner runs in **stop-when-blocked** mode: when its backend's limits block it (Groq's
token bucket, Antigravity's 5-hour or weekly meter, Sonnet without an allowance in the daytime or
at weekly ≥ 75% at night), it stops instead of waiting. It also stops when the queue has no
company left for it. Nothing is lost; the user starts it again with `/extract-run <name>`.
Companies are added with `/extract-add`; a running backend claims the next unclaimed one each time
it finishes a company (one backend per company, in queue order).

## 1. Pre-checks, per named backend (stop for that backend and say why if one fails)

1. Not already running: a runner for backend B is alive if `pgrep -fl "extraction_pipeline.worker --backends"`
   shows a process whose `--backends` list contains B (this also catches older runners started
   with several backends in one process). If alive, do not start a second one; the other named
   backends go ahead.
2. `.venv/bin/python -c "import sec10k"` works (once).
3. Gates now: `.venv/bin/python -m sec10k.extraction_pipeline.router status` (once). Show it as is.
4. Work: `.venv/bin/python -m sec10k.extraction_pipeline.jobs status` (once). If nothing is pending or
   retrying at all, say so, suggest `/extract-add`, and stop. (Whether a company is free for a given
   backend is decided by the runner itself: it stops at once with `backend_idle` if there is none.)

## 2. Sonnet allowance (only for `sonnet PCT`)

`.venv/bin/python -m sec10k.extraction_pipeline.router allow-sonnet <PCT> [--over-weekly]`; show the
limit and expiry. If the 5-hour meter is already ≥ PCT, say the allowance has no effect until the
window resets. Without PCT: grant nothing (Sonnet then runs only at night, 23:00–07:00 IST, while
weekly usage < 75%, or within an allowance still valid from earlier).

## 3. Start one runner per backend, always under caffeinate

Never start a runner without `caffeinate -i` (keeps the Mac from idle-sleeping; closing the lid
still sleeps it). For each backend B to start (`groq`, `antigravity_cli`, `claude_cli`):

```
mkdir -p data/logs data/state
nohup caffeinate -i .venv/bin/python -m sec10k.extraction_pipeline.worker --backends B \
  --stop-when-blocked >> data/logs/extraction_B.stdout 2>&1 &
echo $! > data/state/extraction_runner_B.pid
```

The pid file holds caffeinate's PID; the runner is its child and caffeinate ends when it ends.
After 20 seconds, per runner: if alive, show its last log lines for that backend from
`data/logs/extraction.log`; if it already ended, read its last `backend_stopped` or `backend_idle`
event in `data/logs/extraction_events.jsonl` and report the reason (e.g. no free company, daytime
without allowance). If it died otherwise, show the end of its `.stdout` file.

## 4. Watch and push

If this session is not already watching the events file, start a Monitor on
`tail -n0 -F data/logs/extraction_events.jsonl` (filter: `"push": ?true|Traceback|Error`), re-armed
every 30 minutes while any runner is alive. One Monitor covers all runners (they share the file).
For every line with `"push": true` send a PushNotification: `[<kind>] <backend>: <first line of message>`;
for `summary` and `run_stopped` the whole message (it is short). For `backend_stopped`, say which
limit and the restart command the message names. Stop watching once no runner is alive (check the
`data/state/extraction_runner_*.pid` files and `pgrep`). When the last runner stops, remind the user
that `.venv/bin/python -m sec10k.graph.facts` loads the results into Neo4j once it is up (do not run
it unasked).

## 5. Report

Backends started (with PIDs) and any not started (already running / no work / stopped at once, with
the reason), Sonnet allowance if any, stop-when-blocked, caffeinate on, log paths
(`data/logs/extraction.log`, `data/logs/extraction_events.jsonl`), current gates.
Stopping: `/wind-down <name>` stops one backend's runner; `/wind-down` stops everything.
