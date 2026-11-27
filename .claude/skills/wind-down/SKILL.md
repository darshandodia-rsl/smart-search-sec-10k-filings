---
name: wind-down
description: End the working day on the smart-search-sec-10k-filings project. Gracefully stops any running project jobs (ingestion, embedding, API server), stops every Docker container, and then quits OrbStack itself, so nothing is left running or half-written. Use when the user says "wind down", "end of day", "shut everything down" or "stop all containers".
disable-model-invocation: true
argument-hint: "[groq | agy | sonnet]   (a name = stop only that extraction backend's runner)"
---

# Wind down

Stop everything the project started, in an order that avoids corrupting data. Run commands from the
project root. Say in one line what you are about to stop before stopping it. Never use `kill -9`,
`docker kill`, `docker compose down -v`, or `docker system prune`: they can lose data or work.

## 0. Only one backend (when a name was given)

With `groq`, `agy` or `sonnet` (backends `groq`, `antigravity_cli`, `claude_cli`): stop **only** that
extraction backend's runner and nothing else (no other runners, no Docker, no OrbStack).
Find it with `pgrep -fl "extraction_pipeline.worker --backends"`: the process whose `--backends` list
contains that backend (its pid is also in `data/state/extraction_runner_<backend>.pid`). If that
process also runs other backends (an older runner started with several), say so and ask before
stopping it, since it would stop those too. Stop it as in step 1.3–1.6 (SIGINT, wait, SIGTERM),
report, and end here. If no runner for it is alive, say so.

## 1. Stop project jobs first (they write to Neo4j)

Ingestion and embedding jobs write to Neo4j, so they must stop before the database does.

1. List what is running:
   `pgrep -fl "sec10k|uvicorn"` (macOS `pgrep` uses extended regex by default; do not pass `-E`).
   Project jobs are `python -m sec10k.<module>`; later also a FastAPI/uvicorn server.
   Ignore the shell that is running this very check, this Claude session, and any `claude` processes.
2. If nothing matches, say "no project jobs running" and go to step 2.
3. For each match, ask it to stop gracefully with **SIGINT** (`kill -INT <pid>`), the same as Ctrl+C.
   The pipelines are resumable and idempotent, so a clean interrupt loses nothing.
4. Wait up to 30 seconds, checking every few seconds that the process is gone.
5. If one is still alive, send **SIGTERM** (`kill -TERM <pid>`) and wait another 30 seconds.
6. If it is *still* alive, stop and tell the user the PID and command. Ask before doing anything harsher.

## 2. Stop the Docker containers

1. Stop this project's stack, giving Neo4j time to flush to disk:
   `docker compose stop -t 60`
   (`stop`, not `down`: the container is kept, and the graph lives in `data/neo4j/` either way.)
2. The user asked for *all* containers, so list anything else still running:
   `docker ps --format '{{.Names}}\t{{.Image}}'`
   and stop each with `docker stop -t 30 <name>`. Name every one you stop in the report.
3. If Docker is not running at all, say so and skip this step. Do not start it just to stop things.

## 3. Quit OrbStack

Only after step 2 shows no running containers (`docker ps` empty), free the memory OrbStack holds:

1. `orbctl stop` (no arguments: stops the whole OrbStack service, Docker and any Linux machines).
2. `osascript -e 'tell application "OrbStack" to quit'` to close its menu-bar app.
3. Confirm with `orbctl status`; it should no longer say "Running".
4. If a container would not stop in step 2, do **not** quit OrbStack; tell the user why.
5. If OrbStack was already stopped, say so and move on.

## 4. Verify and report

- `docker ps` shows no running containers, and `pgrep -fl sec10k` finds only the checking shell.
- Report in a short list: what jobs were stopped (and whether SIGINT was enough), which containers
  were stopped, whether OrbStack was quit, and anything that would not stop.
- Remind the user, in one line each (do not do these unasked): whether `git status --short` shows
  uncommitted changes, and that `PLAN.md`'s progress log is updated if a step was finished today.
- State whether OrbStack was quit. `/start-up` starts it again (`open -a OrbStack`).
