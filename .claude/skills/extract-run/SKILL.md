---
name: extract-run
description: Start (or resume) the LLM extraction run on the smart-search-sec-10k-filings project with Groq + Antigravity, and Sonnet only if the user adds --with-sonnet. Optionally adds a wave of companies, starts the background runner(s) under caffeinate in stop-when-blocked mode, and watches events to send phone pushes. Use only when the user invokes /extract-run.
disable-model-invocation: true
argument-hint: "[TICKER ...] [--first-years N] [--with-sonnet [PCT] [--over-weekly]]   (no tickers = resume)"
---

# Extract run

Spends LLM quota. Only ever run because the user typed /extract-run. Run from the project root;
say in one line what you are about to do before each step that changes anything.
This starts **Groq + Antigravity**. **Sonnet only if the user wrote `--with-sonnet`**: then a
separate Sonnet runner is started too (step 4), optionally with a 5-hour allowance of PCT %
(`--over-weekly` only if the user wrote it). Otherwise Sonnet is not touched.
The runners always run in **stop-when-blocked** mode: when a backend's limits block it (Groq's daily
budget, Antigravity's 5-hour or weekly meter), that backend stops instead of waiting, and the
runner exits once all backends have stopped. The Sonnet runner does the same: with no allowance
in the daytime, or at weekly ≥ 75% at night, it stops at once (that is expected, not an error).
Nothing is lost; the user restarts with `/extract-run` (no tickers = resume) once the limits are
restored.

## 1. Pre-checks (stop and tell the user if any fails)

1. No main runner already running: if `data/state/extraction_runner_main.pid` exists and
   `ps -p <pid> -o command=` shows `sec10k.extraction_pipeline.worker`, report it and stop.
   With `--with-sonnet`, check `data/state/extraction_runner_sonnet.pid` the same way; if a Sonnet
   runner is alive, do not start another one (say so; the rest of the run goes ahead).
2. `.venv/bin/python -c "import sec10k"` works.
3. Gates now: `.venv/bin/python -m sec10k.extraction_pipeline.router status`. Show the output as is.

## 2. Add the wave (only if tickers were given)

1. Dry run: `.venv/bin/python -m sec10k.extraction_pipeline.planner <TICKERS> [--first-years N]`;
   show the table.
2. Add: `.venv/bin/python -m sec10k.extraction_pipeline.jobs add <TICKERS> [--first-years N]`.
   Safe to repeat: items already in the store are left untouched. The ticker order is the priority.

With no tickers: run `.venv/bin/python -m sec10k.extraction_pipeline.jobs status`; if nothing is
pending or retrying for Groq/Antigravity, say so and stop.

## 3. Start the runner in the background, always under caffeinate

Every runner start goes through `caffeinate -i`, so the Mac does not idle-sleep during the run
(closing the lid still sleeps it; say so if relevant). Never start the runner without it.

```
mkdir -p data/logs data/state
nohup caffeinate -i .venv/bin/python -m sec10k.extraction_pipeline.worker --backends groq antigravity_cli \
  --stop-when-blocked >> data/logs/extraction_main.stdout 2>&1 &
echo $! > data/state/extraction_runner_main.pid
```

The pid file holds caffeinate's PID; the runner is its child and caffeinate ends when it ends.
After 20 seconds check the process is alive (`ps -p`) and show the last 10 lines of
`data/logs/extraction.log`. If it died, show the end of `data/logs/extraction_main.stdout` and stop.

## 4. Sonnet runner (only with `--with-sonnet`)

1. If PCT was given: `.venv/bin/python -m sec10k.extraction_pipeline.router allow-sonnet <PCT> [--over-weekly]`
   and show the result (limit, expiry).
2. Start the separate Sonnet runner, also under caffeinate and in stop-when-blocked mode:
   ```
   nohup caffeinate -i .venv/bin/python -m sec10k.extraction_pipeline.worker --backends claude_cli \
     --stop-when-blocked >> data/logs/extraction_sonnet.stdout 2>&1 &
   echo $! > data/state/extraction_runner_sonnet.pid
   ```
3. After 20 seconds: if it is alive, show its last `claude_cli` log lines; if it already ended, read
   its `backend_stopped` event and report the reason (e.g. daytime without allowance).

## 5. Watch and push

If this session is not already watching the events file, start a Monitor on
`tail -n0 -F data/logs/extraction_events.jsonl`, re-armed every 30 minutes while a runner is alive.
For every line with `"push": true` send a PushNotification:
`[<kind>] <backend>: <first line of message>`; for `summary` and `run_stopped` send the whole
message (it is short). For `needs_allowance`, end with "reply /extract-sonnet <5-hour %>".
For `backend_stopped` (blocked by a limit), say which limit and that `/extract-run` resumes later.
One Monitor covers both runners (they share the events file). Stop watching once no runner is
alive (check both pid files). When a run stops, remind the user
that `.venv/bin/python -m sec10k.graph.facts` loads the results into Neo4j once it is up (do not
run it unasked).

## 6. Report

Wave added (items, companies), backends started (Groq + Antigravity, plus Sonnet if asked, with its
allowance or stop reason), stop-when-blocked, PIDs, caffeinate on,
log paths (`data/logs/extraction.log`, `data/logs/extraction_events.jsonl`), current gates.
Stopping: `/wind-down` (sends SIGINT; the runner finishes the calls in progress and exits).
