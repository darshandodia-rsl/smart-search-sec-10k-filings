---
name: extract-status
description: Show the state of the LLM extraction run on the smart-search-sec-10k-filings project - which runners are alive, progress, ETA, per-backend usage, gates and anything that needs the user. Read-only. Use when the user invokes /extract-status.
disable-model-invocation: true
---

# Extract status

Read-only: never start, stop or change anything. Run from the project root.

1. Runners: each backend has its own runner, pid file `data/state/extraction_runner_<backend>.pid`
   (`groq`, `antigravity_cli`, `claude_cli`). For every such file, and for any other process from
   `pgrep -fl "extraction_pipeline.worker --backends"`, say which backends it runs, alive or not
   (`ps -p <pid> -o etime=,command=`) and for how long. If the job store
   `data/state/extraction_jobs.sqlite` does not exist, say no run has happened yet and stop.
2. Progress: `.venv/bin/python -m sec10k.extraction_pipeline.progress` (% done, ETA, per backend).
3. Gates now: `.venv/bin/python -m sec10k.extraction_pipeline.router status` (reads the usage meters).
4. Recent activity: last 5 lines of `data/logs/extraction.log`, and any `errors`,
   `needs_allowance` or `backend_stopped` (blocked by a limit) events of the last 24 h in
   `data/logs/extraction_events.jsonl`.
5. Summarize in at most 8 lines. Flag anything that needs the user: Sonnet waiting for an
   allowance, errors in a row, items failed for good, a runner that died, a backend idle with work
   left, or a backend stopped by its limits with work left (say which limit and the restart command,
   `/extract-run groq|agy|sonnet [PCT]`). Queued companies no backend has claimed yet are worth a
   line too: they wait until a backend is started.
