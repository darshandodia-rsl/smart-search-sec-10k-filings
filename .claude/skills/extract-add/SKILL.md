---
name: extract-add
description: Add companies to the LLM extraction queue (pool) on the smart-search-sec-10k-filings project - shows the planner dry run, then adds the work items. Starts nothing and spends no LLM quota. Use only when the user invokes /extract-add.
disable-model-invocation: true
argument-hint: "TICKER [TICKER ...] [--first-years N]"
---

# Extract add

Puts companies in the queue (the "pool"). It never starts a runner: the user starts backends
separately with `/extract-run groq|agy|sonnet`. Run from the project root; say in one line what
you are about to do before each step that changes anything.

How the pool works: companies are handed out in the order they were added. A backend that is
free claims the next unclaimed company and keeps it until all its filings are done (one model per
company). A backend that is **running and still busy** will also claim companies added now; a
backend that already ran out of work has stopped and must be started again with `/extract-run`.

## 1. Check the arguments

At least one ticker is required. If none was given, say so and stop. Tickers are matched against
`data/edgar-software-corpus/companies.csv`; the planner reports any it does not know.

## 2. Dry run

`.venv/bin/python -m sec10k.extraction_pipeline.planner <TICKERS> [--first-years N]`; show the table
as is (it prints a harmless tokenizer length warning; leave it out).

## 3. Add to the queue

`.venv/bin/python -m sec10k.extraction_pipeline.jobs add <TICKERS> [--first-years N]`.
Safe to repeat: items already in the store are left untouched. The ticker order is the priority.

## 4. Report

Items added, per-company pending counts (only the companies just added), and which backends are
running right now (any process matching `sec10k.extraction_pipeline.worker`, with its `--backends`).
If a running backend is busy, say it may claim these companies. If none is running, say the user
starts one with `/extract-run groq|agy|sonnet [PCT]`.
