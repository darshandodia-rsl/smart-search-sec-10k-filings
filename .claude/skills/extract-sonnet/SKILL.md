---
name: extract-sonnet
description: Let the LLM extraction on the smart-search-sec-10k-filings project use Claude Sonnet. Optionally grants a daytime allowance (Sonnet may run until the Claude 5-hour meter reaches the given %), and starts the separate Sonnet runner under caffeinate if it is not running. Use only when the user invokes /extract-sonnet.
disable-model-invocation: true
argument-hint: "[5-hour %] [--over-weekly]   (no % = night rules only)"
---

# Extract with Sonnet

Spends Claude Pro quota. Only ever run because the user typed /extract-sonnet. Run from the project
root; say in one line what you are about to do before each step that changes anything.

This runner **waits** for its window (e.g. start it in the evening and it begins at 23:00). For a
Sonnet runner that stops as soon as it is blocked, use `/extract-run … --with-sonnet [PCT]` instead.

The Sonnet runner obeys the schedule in `CLAUDE.md` on its own: at night (23:00–07:00 IST) it runs
while weekly Claude usage is < 75%; otherwise only within an allowance, and never at weekly ≥ 90%
unless the allowance was granted with `--over-weekly`.

## 1. Allowance (only if a % was given)

1. The % must be a number from 1 to 100. Pass `--over-weekly` **only** if the user wrote it.
2. Say what you will grant, then run:
   `.venv/bin/python -m sec10k.extraction_pipeline.router allow-sonnet <PCT> [--over-weekly]`
3. Show the result (limit, expiry = when the current 5-hour window resets). If the 5-hour meter is
   already ≥ PCT, say the allowance has no effect until the window resets.

With no %: grant nothing. Say Sonnet will only run under the night rule (or a later allowance).

## 2. Start the Sonnet runner if it is not running, always under caffeinate

1. If `data/state/extraction_runner_sonnet.pid` exists and `ps -p <pid> -o command=` shows
   `sec10k.extraction_pipeline.worker`, it is running: an allowance from step 1 takes effect within
   ~5 minutes; skip to step 3.
2. Otherwise check there is work: `.venv/bin/python -m sec10k.extraction_pipeline.jobs status`
   (pending/retrying items in companies not owned by another backend). If none, say so and stop.
3. Start it. Every runner start goes through `caffeinate -i`; never start one without it.
   ```
   mkdir -p data/logs data/state
   nohup caffeinate -i .venv/bin/python -m sec10k.extraction_pipeline.worker --backends claude_cli \
     >> data/logs/extraction_sonnet.stdout 2>&1 &
   echo $! > data/state/extraction_runner_sonnet.pid
   ```
   After 20 seconds check it is alive and show the last lines of `data/logs/extraction.log` that
   start with `claude_cli`. If it died, show the end of `data/logs/extraction_sonnet.stdout` and stop.
   The Sonnet runner only recovers its own interrupted items, so it is safe next to the main runner.

## 3. Watch and push

If this session is not already watching the events file, set up the Monitor and pushes exactly as
in step 4 of the `extract-run` skill (`.claude/skills/extract-run/SKILL.md`).

## 4. Report

Allowance (or none), current Claude meters (`router status`), whether the Sonnet runner was started
or already running (PID), caffeinate on. Stopping: `/wind-down`.
