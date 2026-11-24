# GraphRAG over SEC 10-K filings: Phase Plan

Each phase ends with something you can **run and see**. Don't start a phase until the previous
one works.

## The core problem in one paragraph
Having an LLM read all ~55M tokens once would take ~9 months on the free API (200K tokens/day),
and weeks even with the Claude workaround. So the plan is built around **spending LLM tokens only
where they add value**:
- Do everything that doesn't need an LLM (structure, chunking, embeddings) without one.
- Skip text we've already processed. 10-Ks repeat a lot from year to year, so we only send
  *new or changed* paragraphs to the LLM.
- Make the pipeline resumable and budgeted, so it can run for weeks without wasting quota.

---

## Phase 0: Setup
**Goal:** a working environment and a "hello world" for every component.
- git repo, Python environment, project folder layout
- Docker + Neo4j running; open the Neo4j browser
- Local embedding model loads and embeds a sentence
- **LLM provider interface** with two backends: gpt-oss (free API) and Claude (`claude -p`).
  Both answer the same test prompt.

**You'll learn:** why we put LLMs behind an interface; how `claude -p` works as a backend.

> **Finding (2026-11-24):** `claude -p` works on the Pro plan. But a 2-token prompt used ~26K
> tokens of overhead (Claude Code's system prompt + tools + our CLAUDE.md), and it defaulted to Opus.
> In Phase 0, test: a cheaper model (`--model`), replacing the system prompt, disabling tools,
> running from a folder without CLAUDE.md, and batching several chunks per call. Check flags
> with `claude --help`.

## Phase 1: Know the data
**Goal:** understand the corpus before designing anything.
- Load all filings, check section quality (empty or broken sections, leftover HTML, tables)
- Measure: tokens per section, per company, per year
- **Measure year-over-year repetition** (how much of a 2023 risk section also appears in 2022).
  This number tells us how much LLM budget the "only process changes" trick saves.

**You'll learn:** why data profiling comes first and drives the cost estimates.

## Phase 2: Structural graph + vector baseline (no LLM)
**Goal:** a working plain RAG system, used as the baseline to compare against.
- Structural graph straight from metadata: `Company → Filing → Section → Chunk`
- Chunking strategy (section-aware, sized for both embedding and extraction)
- Embed every chunk locally and store in a Neo4j vector index
- Simple vector-search Q&A

**You'll learn:** chunking trade-offs; how much you get "for free" without any LLM extraction.

## Phase 3: Graph schema + extraction prototype
**Goal:** decide *what* the graph holds, and test LLM extraction on a few filings.
- Schema built for your question types: relationships (competitors, suppliers, customers,
  partners), risks, products/segments and themes, all tagged with the fiscal year
  (for change over time)
- **Cheap extraction tiers first (no LLM), run on every chunk:**
  - *Tier 2, pattern/dictionary:* match our 130 known companies (names, tickers) and
    phrases like "competitors include…" → `MENTIONS` edges
  - *Tier 3, local NLP model* (e.g. spaCy or GLiNER): find other companies, products and places
- **Use tiers 2–3 as a filter:** only chunks likely to hold a relationship (e.g. 2+ entities)
  go to the LLM; boilerplate is skipped. Measure how much LLM usage this saves.
- Extraction prompt with a strict output format (schema-guided), using LlamaIndex
- Run it on ~3 filings, inspect the results by hand, iterate on the prompt
- Measure **tokens per chunk** to get a real full-corpus cost estimate

**You'll learn:** schema-guided vs free-form extraction; why the schema follows from the questions;
the tiered "cheapest method that's good enough" approach (structure → patterns → local NLP → LLM).

## Phase 4: Production ingestion pipeline
**Goal:** the full-corpus pipeline, built properly.
- Job state tracking (which chunks are done, failed or pending) so it is **resumable**
- **Rate limiter + daily token budget** for each backend
- **Caching** of LLM responses (never pay twice for the same chunk)
- **Change detection**: skip unchanged paragraphs from year to year
- Batching several chunks per request where it saves tokens
- Retries, failure logging, progress reporting
- First small wave: ~5 companies × 2 years

**You'll learn:** idempotency, backpressure, and how real ingestion survives crashes and limits.

## Phase 5: Entity resolution
**Goal:** one node per real-world thing.
- "NVIDIA", "NVIDIA Corporation" and "Nvidia Corp." → one entity; link to `companies.csv` (CIK/ticker)
- Merge duplicate risks and themes using embeddings and rules

**You'll learn:** why entity resolution makes or breaks GraphRAG quality.

## Phase 6: Retrieval
**Goal:** answer all four question types.
- Hybrid retrieval: vector search + graph traversal + text-to-Cypher
- Time-aware queries (compare years)
- Cross-company queries (aggregate across the graph)
- Answers cite their sources (company, year, section)
- Compare against the Phase 2 baseline on a handful of questions

**You'll learn:** when graph retrieval beats vector search, and when it doesn't.

## Phase 7: FastAPI service
**Goal:** deployable API.
- `/query` endpoint, `/health`, ingestion status endpoint
- Config via environment variables, structured logging

## Phase 8: Full-corpus run (ongoing)
**Goal:** process all 1,018 filings in waves until done.
- Run waves as quota allows, monitor progress, spot-check quality

---

## Open questions (decide when we get there)
- Evaluation suite: not in scope for now; possibly add later.

**Resolved (2026-11-24):** Python 3.12 in one root `.venv`; code in `src/`, corpus in
`data/edgar-software-corpus/`; free API = Groq.

---

## Progress log
| Date | Phase.Step | What was done | Open items |
|---|---|---|---|
| 2026-11-24 | Planning | Asked clarifying questions; wrote CLAUDE.md and PLAN.md | — |
| 2026-11-24 | Planning | Tested `claude -p` on Pro: works, ~26K tokens overhead/call | Reduce overhead in Phase 0 |
| 2026-11-24 | Planning | Added extraction tiers 2–3 (patterns + local NLP) to Phase 3 as an LLM pre-filter | — |
| 2026-11-24 | 0 (partial) | Moved corpus to `data/`, created `src/`, one Python 3.12 `.venv` with crawler deps (unpinned), fixed crawler `config.json` paths, removed old 3.9 venv | git, Docker/Neo4j, embeddings, LLM interface |
| 2026-11-24 | 0.1 git | `git init` + `.gitignore` (excludes `.venv/`, `data/`, `.env`, caches). Not committed yet | Docker not installed — needed for Neo4j |
| 2026-11-24 | 0.2 Neo4j | OrbStack (Docker). `docker-compose.yml`: neo4j 5.26 community + APOC, 1G heap/1G pagecache, data in `data/neo4j/`. Password in `.env`. Verified via cypher-shell (~1.3 GB RAM idle) | Embeddings, LLM interface |
| 2026-11-24 | 0.3 Embeddings | Installed `llama-index-embeddings-huggingface` (torch 2.15, sentence-transformers 6.3). `BAAI/bge-base-en-v1.5` runs on MPS: 768-dim; sim(related)=0.88 vs sim(unrelated)=0.48; ~23 chunks/s at ~512 tokens → full corpus ≈ 1.5 h | LLM interface |
| 2026-11-24 | 0.4 LLM interface | `src/llm.py`: `LLMProvider.complete()` + `GroqProvider` (gpt-oss-120b, reasoning_effort=low) + `ClaudeCLIProvider` (WORKAROUND; haiku, `--system-prompt`, `--tools ""`, `--strict-mcp-config`, temp cwd). Minimal guards: SDK retries, RPM pacing, persisted daily token budget (`data/state/`). Both answered the same prompt. `claude -p` overhead cut from ~26K → ~640 input tokens/call. Installed `openai`, `python-dotenv` | No tokens-per-minute (8K TPM) guard yet → Phase 4. Phase 0 commit pending |
| 2026-11-24 | 0.5 deps | `requirements.txt`: direct deps pinned with `==` (no full lock file; crawler deps excluded). `.env.example` template with all settings, no secrets. Short `README.md` (what, stack, setup, layout) | Phase 0 commit pending |
| 2026-11-24 | 0.6 Reproducible corpus | `corpus/`: copies of build scripts (UA from `SEC_USER_AGENT` in `.env`), new `step2_select.py` codifying the manual selection (HUBS float ÷1000; DRTTF/ZPTA/GEG excluded; top 130 by float), `setup_crawler.sh` (edgar-crawler @84a8d0c + throttle patch + config template), pinned `corpus/requirements.txt`, frozen `snapshot/`. Verified byte-identical: step2 outputs, step3 manifest/grid/failures, crawler files; 3-filing re-extraction identical in Items 1/1A/7/7A (metadata NaN vs None only). README "Rebuild the corpus" section | Full re-download not run (would re-fetch ~5 GB from SEC). Corpus is 950 filings (FY2015–24), not 1,018 as CLAUDE.md says |
| 2026-11-24 | 0.7 Docs | CLAUDE.md: 950 filings (FY2015–24), `corpus/` in layout, manifest = authoritative filing list, crawler deps pinned in `corpus/requirements.txt`. `.gitignore`: `data/` → `/data/` (top-level only) | Phase 0 commit pending |
