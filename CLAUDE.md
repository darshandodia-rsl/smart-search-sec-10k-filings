# Instructions for Claude sessions in this project

Read this file and `PLAN.md` at the start of every session. `PLAN.md` holds the phases and the
progress log. This file covers *how* to work.

## What this project is
A GraphRAG system over SEC 10-K filings (130 US tech companies, fiscal years 2015–2024,
950 filings, ~38M tokens). It is a learning project built the way production teams would build it:
not a toy demo, but also not full enterprise hardening.

## Who you're working with
The user knows GraphRAG basics and is here to **learn by building**. Understanding matters more
than speed.

## How to work (most important)
1. **One small step at a time.** Do one step of the current phase, explain it, then stop and wait
   for the user's OK before the next step.
2. **No unrequested work.** No extra refactors, "while I'm here" optimizations, extra features,
   extra files or extra dependencies. If you spot something worth doing, *mention it in one line*
   and let the user decide.
3. **Say what you'll do before doing it.** Before you edit code or run a command that changes
   anything, say in 1–2 sentences what you're about to do and why.
4. **Always explain the "why".** For every design choice, give the reason and, when it helps,
   what production would do differently and why we aren't.
5. **Keep explanations simple and short.** Use plain language, short paragraphs, and an example or
   analogy where it helps. No info-dumps: give the key idea, then offer to go deeper.
   Define jargon the first time you use it.
6. **Don't silently change decisions.** The decisions below are settled. If one looks wrong,
   raise it and ask. Don't work around it.
7. **Be honest about results.** If something failed, was skipped or is only partly working,
   say so plainly.
8. **Update the progress log** in `PLAN.md` at the end of each completed step: date, what was
   done, and anything still open.
9. **Code comments (every file you generate).**
   - **Doc comments everywhere:** every file (module docstring), class and method/function gets one.
   - **Plain language:** write them simply, as if explaining to someone new to the code.
   - **Say what and why:** each doc comment says what the code does *and* why it is needed in the
     overall project (where it fits in the pipeline, what depends on it).
   - **Fresh perspective only:** describe the code as it is now. No history or trade-off
     stories ("because X couldn't be done, we did Z", "changed from…", "previously…").
     That belongs in `PLAN.md`'s progress log, not in the code.

## Settled decisions (don't change without asking)
| Area | Decision | Why |
|---|---|---|
| Framework | **LlamaIndex** | Built around ingestion and indexing; has PropertyGraphIndex and a Neo4j integration |
| Graph DB | **Neo4j Community in Docker** | Industry standard, Cypher, built-in vector index, visual browser |
| Embeddings | **Local model** (e.g. BAAI/bge family) on the M3 | Free and unlimited; self-hosted embeddings are common in production |
| LLM access | **Swappable LLM backend** behind one interface | Production code shouldn't depend on one vendor |
| ↳ backend A | `openai/gpt-oss-120b` on **Groq** (free tier, OpenAI-compatible API) | The "real" production path |
| ↳ backend B | Claude **Sonnet** via `claude -p` (Claude Code headless, $20 Pro plan) | **Workaround** for free-API limits. Keep it clearly labeled as a workaround in code. Sonnet, not Haiku: extraction quality should be close to gpt-oss-120b |
| ↳ backend C | Gemini (`gemini-4-hydrogen-medium`) via Google Antigravity `agy` (headless) | Second **workaround** with its own 5-hour + weekly quota, separate from Claude's. Labeled as a workaround in code |
| Backend order | The three backends run **in parallel**, each on **whole companies** (one model per company). Preference **Groq > Antigravity > Sonnet** (Sonnet only within its schedule) | Quotas are separate, so parallel uses time-limited windows (Sonnet's night) fully; one model per company keeps year-over-year changes from being model artifacts. Groq is the production path; Sonnet shares the Pro plan the user needs for Claude Code |
| Sonnet schedule | **Night (23:00–07:00 IST) and weekly Claude usage < 75%:** may use the full 5-hour window, weekly usage re-checked between batches (stop at 75%). **Daytime:** only within an allowance the user grants with `/extract-run sonnet <session %>`; without one the Sonnet runner stops at once (phone push says how to restart); Groq + Antigravity keep running meanwhile. **Weekly ≥ 90%: no Sonnet even with an allowance**, unless the user explicitly overrides | Keeps the Pro plan usable for the user's own work |
| Antigravity schedule | Runs any time; pauses at **5-hour meter ≥ 95%** until it resets, stops for the week at **weekly ≥ 90%**. The user uses Antigravity for nothing else | Uses its separate quota fully, with a little headroom |
| Schema/prompt freeze | **Frozen since 2026-11-26 at schema v3 / prompt v6** (after wave 1, PANW CRWD TENB RPD QLYS first 2 years, was checked on v6). **No schema or prompt change without the user's explicit OK.** Issues found later are handled without re-extraction where possible (Phase 5 entity resolution, retrieval filters, evidence-based relabel rules in code); a needed change = version bump + selective re-extraction (`jobs replan --requeue --route`) | Every change after the big run starts means re-paying for finished work; tune on a small sample, then freeze, as production pipelines do |
| Company assignment | Taxi rank: a free backend claims the next unassigned company (priority order) and keeps it until all its years are done | Faster backends take more companies; one model per company |
| Starting runs | **Never start an extraction run on your own**, on any backend, at any time (night included). Ask; start only when the user says so, through the user-invoked skills: `/extract-add TICKERS` puts companies in the queue (starts nothing); `/extract-run groq|agy|sonnet [PCT]` starts **exactly the named backends, one runner process each**, all **stop-when-blocked** (a backend blocked by its limits, or with no free company left, stops instead of waiting; the user restarts it later); Sonnet only when the user names it; `/wind-down <name>` stops one backend; `/extract-status` (read-only). **Every runner start is wrapped in `caffeinate -i`** | Runs spend shared quotas and heat the laptop; caffeinate keeps an unattended run from idle-sleeping |
| Scope | Pipeline built for the **full corpus**, run in waves until everything is done | Shows how production ingestion works: resumable, budgeted, idempotent |
| Interface | **FastAPI** service | How it would be deployed |
| Version control | **git**, one commit per completed step (only when the user says to commit) | So each step can be reviewed later |

## Hard constraints to always respect
- **Free API limits (gpt-oss-120b):** 30 requests/min, 1K requests/day, 8K tokens/min,
  **200K tokens/day**. Any code that calls it must have rate limiting, retries and a daily token
  budget.
- **Claude Pro limits:** usage resets every 5 hours and is not unlimited. Treat it as a budget too.
  Meters: `claude -p /usage` (5-hour session and weekly %).
- **Antigravity limits:** separate 5-hour and weekly quotas (roughly 0.6M filing-text tokens/week,
  measured coarsely). Meters: `agy -p /usage`.
- **Machine:** Apple M3, 16 GB RAM. Watch memory use (Neo4j + embedding model + Python together).
- **Never re-pay for work already done:** LLM calls must be cached and the pipeline must be
  resumable. Losing a day's quota to a crash is a bug.

## Project layout
```
smart-search-sec-10k-filings/      ← project root (run commands from here)
├── CLAUDE.md, PLAN.md, README.md
├── .venv/                         ← the ONE virtual env (Python 3.12) for everything
├── pyproject.toml                 ← makes src/sec10k an installable package (pip install -e .)
├── src/sec10k/                    ← all GraphRAG code: ONE package, run with `python -m sec10k.<module>`
│   ├── config.py                  ← shared paths/settings; never compute paths from __file__ elsewhere
│   ├── llm/                       ← LLM backends
│   ├── embeddings.py              ← the one embedding model (chunker tokenizer must match it)
│   ├── ingest/                    ← pipeline: loader, headings, chunker, embed
│   ├── graph/                     ← Neo4j store + graph writers (retrieval/, api/ later)
│   ├── extract/                   ← extraction tiers: alias/pattern matching, local NLP, LLM
│   ├── extraction_pipeline/       ← runs LLM extraction at scale: planner, job store, batcher, router, worker
│   └── analysis/                  ← one-off studies; may import the pipeline, never the reverse
├── corpus/                        ← scripts + frozen snapshot to rebuild the corpus (in git)
└── data/                          ← git-ignored
    └── edgar-software-corpus/     ← downloaded corpus + crawler (treat as read-only)
```
- Use `.venv/bin/python` / `.venv/bin/pip`. Don't create other venvs.
- The edgar-crawler's own pins don't support Python 3.12; the versions verified to reproduce the
  corpus are pinned in `corpus/requirements.txt`. How to rebuild the corpus: README.md.
- Secrets live in `.env` (git-ignored); `.env.example` lists every setting.

## Data files (already exist, don't modify)
All paths are relative to `data/edgar-software-corpus/`.
- `data/extracted/10-K/*.json`: one filing per file. Sections `item_1` (Business), `item_1A`
  (Risk Factors), `item_7` (MD&A) and `item_7A` (Market Risk), plus metadata (cik, company,
  filing_date, period_of_report, sic, ...).
- `manifest.csv`: **the authoritative filing list** (950 rows: FY2015–2024, one per company-year)
  with word and token estimates. `data/extracted/` holds 1,018 files; the 68 extra ones are outside
  the year range, so always select filings via the manifest.
- `companies.csv`: the 130 companies (rank, cik, ticker, name, sic).
- `data/raw`, `data/indices`, `edgar-crawler/` and `scripts/` came from the download step.
