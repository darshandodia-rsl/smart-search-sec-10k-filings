---
name: start-up
description: Start the working day on the smart-search-sec-10k-filings project. Makes sure Docker is up, starts the Neo4j container and waits until it answers, checks the Python environment and today's LLM budgets, and says where the project was left off. Use when the user says "start up", "start the day", "get everything running" or "boot the project".
disable-model-invocation: true
---

# Start up

Bring the project to a ready-to-work state. Run commands from the project root. Say in one line
what you are about to start before starting it. Do **not** start ingestion, embedding or other
long jobs: they use LLM budget and memory, and the user decides when to run them.

## 1. Docker

1. `docker info >/dev/null 2>&1`. If it fails, Docker (OrbStack) is not running: run
   `open -a OrbStack`, then poll `docker info` every 3 seconds for up to 60 seconds.
   If it still is not up, stop and tell the user.
2. Check `.env` exists and sets `NEO4J_PASSWORD` (`grep -q '^NEO4J_PASSWORD=' .env`).
   If not, stop and tell the user to create it from `.env.example`. Never invent or print the password.

## 2. Neo4j

1. `docker compose up -d` (starts the `neo4j-sec10k` container; its data in `data/neo4j/` is kept).
2. Wait until the database answers, up to 90 seconds, checking every 5 seconds. Load the password
   from `.env` without printing it:
   ```
   set -a; . ./.env; set +a
   docker exec -e NEO4J_PASSWORD neo4j-sec10k sh -c 'cypher-shell -u neo4j -p "$NEO4J_PASSWORD" "RETURN 1"'
   ```
3. Once it answers, count what is stored (same command, with
   `"MATCH (n) RETURN count(n) AS nodes"`) so the user can see the graph survived the restart.
4. If it never answers, show the last lines of `docker compose logs --tail 30 neo4j` and stop.

## 3. Python environment

Run `.venv/bin/python -c "import sec10k; print('sec10k ok')"`. If it fails, tell the user and
suggest `.venv/bin/pip install -e .`; do not install anything unasked.

## 4. Today's budgets

Read `data/state/budget_groq.json` and `data/state/budget_claude_cli.json` if they exist. Each holds
`date`, `tokens_used`, `requests`. If `date` is not today, the budget has reset: report it as unused.
Report used against the limits in `CLAUDE.md` (Groq 200,000 tokens/day and 1,000 requests/day;
Claude CLI self-cap 2,000,000 tokens/day).

## 5. Where we left off

Read `PLAN.md` and `COMEBACK_LATER.md` (if present). In three lines or fewer: the current phase,
the last row of the progress log, and the next step. Also show `git status -sb` in one line.

## 6. Report

End with a short ready-check:

```
Docker ✓   Neo4j ✓ (browser http://localhost:7474, N nodes)   Python ✓
Budgets: Groq X / 200,000   Claude Y / 2,000,000
Left off: <one line>.  Next: <one line>.
```

Mark anything that failed with ✗ and say what is needed. Then stop and wait for the user.
