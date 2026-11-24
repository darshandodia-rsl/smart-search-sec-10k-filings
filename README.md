# Smart Search over SEC 10-K Filings (GraphRAG)

A GraphRAG system over 10-K filings of 130 US tech companies (fiscal years 2015–2024,
950 filings, ~38M tokens), built with LlamaIndex and Neo4j. It is a learning project, built the way a
production team would: resumable, budgeted ingestion and swappable LLM backends.

**Status:** Phase 0 (setup) done. See [`PLAN.md`](PLAN.md) for phases and the progress log.

## Stack
- **Graph + vector store:** Neo4j 5.26 Community (Docker)
- **Embeddings:** `BAAI/bge-base-en-v1.5`, run locally (Apple Silicon GPU via MPS)
- **LLM:** swappable backend (`src/sec10k/llm/providers.py`)
  - Groq `openai/gpt-oss-120b` (free tier)
  - Claude via `claude -p` (a workaround for free-tier limits)

## Setup
Requires Python 3.12, Docker (e.g. OrbStack), and optionally the `claude` CLI.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .     # our code as the editable package `sec10k` (src/sec10k/)

cp .env.example .env          # then fill in NEO4J_PASSWORD and GROQ_API_KEY
docker compose up -d          # Neo4j browser: http://localhost:7474 (connect to bolt://localhost:7687)

.venv/bin/python -m sec10k.llm.providers   # smoke test: same prompt to both LLM backends
```

## Layout
```
src/sec10k/            GraphRAG code, one Python package (run: .venv/bin/python -m sec10k.<module>)
├── config.py            shared paths and settings (the only place that knows where files live)
├── llm/
│   └── providers.py     swappable LLM backends (Groq gpt-oss-120b, Claude via `claude -p`)
├── ingest/              pipeline
│   ├── loader.py          load + clean filings, selected via manifest.csv
│   ├── headings.py        recover section headings
│   └── chunker.py         split sections into chunks
└── analysis/            one-off studies, not part of the pipeline
    ├── profile_corpus.py  corpus profiling
    └── compare_chunking.py  chunking comparison

corpus/                scripts + frozen snapshot to rebuild the corpus (see below)
├── step1_candidates.py, step1b_extras.py, step2_select.py   company selection
├── step3_manifest.py    builds manifest.csv
├── spot_check.py        by-eye check of extractions
├── setup_crawler.sh     sets up edgar-crawler (+ throttle patch, config template)
├── requirements.txt     pinned versions that reproduce the corpus
└── snapshot/            frozen company list and build outputs

data/                  corpus + Neo4j data + pipeline state (git-ignored)
docker-compose.yml     Neo4j
pyproject.toml         makes src/sec10k installable; versions stay pinned in requirements.txt
requirements.txt       pinned dependencies for the project venv
.env.example           every setting; copy to .env (git-ignored) for secrets
PLAN.md                phases and progress log
CLAUDE.md              working rules for Claude Code sessions
```

## Rebuild the corpus
The corpus (~5 GB, in `data/edgar-software-corpus/`) is not in git; it is rebuilt from SEC EDGAR.
Filings never change once filed, so freezing the **company list** is enough to get the same corpus.
That frozen list and the build outputs are in `corpus/snapshot/`.

**One-time setup** (set `SEC_USER_AGENT` in `.env` first; SEC requires a name and email):
```bash
.venv/bin/pip install -r corpus/requirements.txt
corpus/setup_crawler.sh        # edgar-crawler at a pinned commit + our rate-limit patch + config.json
```

**A. Exact rebuild from the frozen company list (recommended)**
```bash
cp corpus/snapshot/companies.csv data/edgar-software-corpus/
cp corpus/snapshot/ciks.txt      data/edgar-software-corpus/scripts/

cd data/edgar-software-corpus/edgar-crawler
../../../.venv/bin/python download_filings.py   # 10-Ks 2015–2025, throttled to 8 req/s
../../../.venv/bin/python extract_items.py      # Items 1, 1A, 7, 7A -> JSON (1,018 filings)
cd ../../..

.venv/bin/python corpus/step3_manifest.py       # keep FY2015–2024, dedupe -> manifest.csv (950)
cmp data/edgar-software-corpus/manifest.csv corpus/snapshot/manifest.csv && echo "corpus matches snapshot"
```

**B. Re-run the company selection from scratch** (replaces the two `cp` lines in A)
```bash
.venv/bin/python corpus/step1_candidates.py  # all SEC tickers -> software SIC 7370–7374 with an FY2024 10-K
.venv/bin/python corpus/step1b_extras.py     # adds PANW, FTNT, AAPL (hardware SIC codes)
.venv/bin/python corpus/step2_select.py      # fixes known data errors, ranks by public float, keeps top 130
```
SEC's live data (tickers, floats, new filings) keeps changing, so route B will **not** give
exactly the same 130 companies today; check the floats for new data errors as `step2_select.py`
does. To reproduce the original selection, copy `corpus/snapshot/step1_raw.csv` and `extra.csv`
into `data/edgar-software-corpus/scripts/` and run only `step2_select.py`.

`corpus/spot_check.py` prints 5 random extractions next to the raw filing, for a by-eye check.
