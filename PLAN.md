# GraphRAG over SEC 10-K filings: Phase Plan

Each phase ends with something you can **run and see**. Don't start a phase until the previous
one works.

## The core problem in one paragraph
Having an LLM read all ~38M tokens once would take ~6 months on the free API (200K tokens/day),
and weeks even with the Claude workaround. So the plan is built around **spending LLM tokens only
where they add value**:
- Do everything that doesn't need an LLM (structure, chunking, embeddings) without one.
- Skip text we've already processed. 10-Ks repeat a lot from year to year, so we only send
  *new or changed* paragraphs to the LLM.
- Make the pipeline resumable and budgeted, so it can run for weeks without wasting quota.

---

## Phase 0: Setup ✅ Completed 2026-11-24
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

## Phase 1: Know the data ✅ Completed 2026-11-24
**Goal:** understand the corpus before designing anything.
- Load all filings, check section quality (empty or broken sections, leftover HTML, tables)
- Measure: tokens per section, per company, per year
- **Measure year-over-year repetition** (how much of a 2023 risk section also appears in 2022).
  This number tells us how much LLM budget the "only process changes" trick saves.

**You'll learn:** why data profiling comes first and drives the cost estimates.

> **Results** (`src/sec10k/analysis/profile_corpus.py`, details in `data/profile/`):
> - Quality: 3 broken sections (EVTC 2023 Item 1, ALRM 2015 Item 7A, DBX 2022 Item 1A) → trim/skip
>   in the Phase 2 loader. Page footers in ~5% of sections, flattened tables in 17 MD&As. No HTML.
> - Size: **37.7M tokens** (gpt-oss tokenizer). Item 1A = 49%, Item 7 = 29%, Item 1 = 20%.
> - Repetition vs previous filing: 33% at paragraph level, **57% at sentence level**.
> - Cost: Claude tokenizer = 1.62× gpt-oss. Days to finish with both backends: ~21–30 without a
>   filter, **~6–15** if the Phase 3 filter keeps 30–50% of chunks (filter share, prompt/output sizes
>   and real Claude Pro capacity still to be measured).

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
**Goal:** process all 950 filings (from `manifest.csv`) in waves until done.
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
| 2026-11-24 | 1.1 Section quality | `src/profile_corpus.py` checks all 3,800 sections (950 filings × 4) and writes `data/profile/section_quality.csv`. Hand-checked every flag. **Real extraction errors: 3**: EVTC 2023 Item 1 is heading-only; ALRM 2015 Item 7A ran through Item 8–15 (29K words); DBX 2022 Item 1A ran through Items 2–16 (~half of its 50K words). Minor: PD 2019–23 Item 1 starts with a table of contents. **Not errors:** 16 Item 7A sections with no content (cross-reference to MD&A, or "smaller reporting company"); length jumps for DJT/CCC/U/APLD/PEGA look like real business changes. **Noise to clean in Phase 2:** page footers/headers in ~5% of sections (208); flattened tables (one number per line) in 17 MD&A sections (ALRM, PCTY, GDDY, BL, ZETA). No HTML leftovers. Fixed Phase 8 count to 950 | **Decided:** Phase 2 loader trims sections at a later "ITEM N" heading and skips EVTC 2023 Item 1; corpus stays read-only |
| 2026-11-24 | 1.2 Token counts | Added real token counts (`o200k_harmony`, gpt-oss's own tokenizer) to `src/profile_corpus.py`; writes `tokens_by_company.csv`, `tokens_by_year.csv`. **Corpus = 37.7M tokens** (manifest estimate 41.6M; 1.20 tokens/word). Item 1A = 49%, Item 7 = 29%, Item 1 = 20%, Item 7A = 2%. Median filing 39K tokens (5–95%: 22K–59K). FY2015 2.2M → FY2024 5.6M (more filings and longer risk sections). Raw read on Groq alone ≈ 189 days of the 200K/day budget, before prompt/output overhead | Updated ~55M → ~38M (~6 months) in CLAUDE.md, PLAN.md, README.md. Claude tokenizer ratio not measured yet |
| 2026-11-24 | 1.3 YoY repetition | `src/profile_corpus.py` compares each section with the same company's previous filing (page/table noise removed), in tokens, writes `repetition_by_section.csv`. Paragraph-level exact match: **33%** repeated (1A 39%, 7 24%). Sentence-level exact: **57%** (1A 66%, 7 43%) — companies keep most text but edit a sentence or two per paragraph, which breaks whole-paragraph matches. Masking numbers adds only ~3 pts. Stable across years (55–60% sentence level); per company 46–67% (10–90%). LLM must read once: 27.1M tokens (paragraph) vs **19.3M (sentence) ≈ 96 Groq-days** input only. Diagnostic: ADBE 2023 1A only 13% reused (real rewrite) | Which unit to skip at (paragraph vs sentence-with-context) → Phase 4 design. Not tried: fuzzy near-duplicate matching, matching against all earlier years |
| 2026-11-24 | 1.4 Cost estimate | Measured tokenizer ratios with one 3K-token sample: Groq = 1.00× o200k (+84 tokens/call); **Claude Haiku = 1.62×** (+621 tokens/call). `report_cost_estimate()` in `src/profile_corpus.py`: measured values + clearly labeled assumptions (4K content/request, 1K prompt, output = 25% of content, 300 reasoning tokens). Overhead means only ~62% of Groq's budget is filing text → **125K text tokens/day (Groq), 774K (Claude at the 2M/day self-cap), 899K both**. Days to finish, both backends: 30 (skip unchanged paragraphs, no filter) → 21 (sentences) → **~6–11** if the Phase 3 filter keeps 30–50%. Groq alone: 46–216 days. **Phase 1 complete** | Claude ratio is one sample; real Pro capacity per 5-hour window unknown (2M/day is our own cap) → measure in Phase 4. Prompt/output assumptions → measure in Phase 3. Phase 1 commit pending |
| 2026-11-24 | 2.1 Loader | `src/loader.py`: `iter_clean_sections()` reads filings via manifest, cleans in memory → `Section` records. Fixes: drop leading TOC (PD 2019–23 Item 1), trim at a later item heading or a *single* "Index to Financial Statements" line (DBX 2022 1A 50.5K→22.2K words; ALRM 2015 7A 29K→298; DOCS 2022–24 7A tail), drop page furniture (2,205 lines incl. Oracle's repeated per-page index link) and **rejoin 410 sentences split by page breaks**, drop 6,393 number-only table lines. Skip < 50 words: 17 sections (EVTC 2023 Item 1 + 16 Item 7A cross-references). 3,783 sections kept, 31.24M words | Flattened-table *labels* stay as short orphan lines |
| 2026-11-24 | 2.2a Chunking test | `src/compare_chunking.py`: paragraph (`SentenceSplitter`, 400 bge tokens, no overlap) vs semantic (`SemanticSplitterNodeParser`, p95) on MSFT FY22+23, CRM FY23 (110K bge tokens). Paragraph: 336 chunks, median 348 tokens, 0 truncated, 0.6 s. Semantic: 194 chunks, median 447, p95 1,876, **83 chunks (43%) over 512 → 44K tokens (40%) silently truncated**, 11 tiny, 42 s (≈4 h for the corpus), boundaries ignore risk headings. YoY identical chunks: paragraph 3%, semantic 10% — **both too unstable for chunk-level change detection** | Decide chunker; change detection must work on paragraphs/sentences, not chunks (Phase 4) |
| 2026-11-24 | 2.2a Heading lookup | Decided: structure-aware paragraph chunking (no semantic chunking: headings already separate topics). Formatting is lost in the extracted text but present in the raw HTML, so `src/headings.py` parses each raw `.htm` (lxml, 6 processes, **17 s for 950 filings**) and caches emphasized blocks (bold/italic/underline via tags or inline CSS; outside tables) per filing in `data/headings/` (39 MB, resumable). Two kinds: `full` (whole block emphasized = heading/risk title) and `lead` (run-in title, "**Competition.** We ..."). Lines match by first 100 normalized chars. Corpus not re-extracted, so Phase 1 stays valid. Fixed 2 bugs found by checking weak filings: wrong text order for run-in titles (ORCL/PEGA) and HTML comments read as text (TWLO/TTWO). **Result:** heading lines found in 99.5% of Item 1A sections (100% of Items 1 and 7); Item 1A 49.9K heading lines / 240K lines; most filings 10–40% heading lines. Random samples: nearly all are real headings/titles | Misses: FDS 2016–18 (titles only marked by a larger font size), DJT 2021–22 (SPAC filings, no emphasis) → fall back to plain packing. Minor false positives (italic footnotes like "Note: …"); harmless, they just start a chunk early |
| 2026-11-24 | 2.2b Chunker (small set) | `src/chunker.py`: paragraph packing ≤ 400 bge tokens; a heading or run-in title always starts a new chunk (consecutive headings stay together); paragraphs > 400 split at sentences (`SentenceSplitter`); each chunk keeps its nearest heading. `chunk_id = {accession}:{section}:{seq}` (position, unique) + `content_hash` (sha1, for "seen before?" checks, not an ID, or identical chunks of 2 filings would merge). `embed_text()` = "COMPANY (TICKER) 10-K FYyyyy \| Item … \| heading" + text. Loader additions found while testing: drop running headers ("PART I", bare "Item 1A", "2023 Form 10-K", company name repeated 2+× per section, `$`/`%`/`____` lines) and join lone bullets "•" to their text (38.9K lines in 1,086 sections); audited every new rule's removals, all furniture. **Test set MSFT/CRM/ORCL/TWLO/FDS (49 filings, 1.93M tokens):** 8,482 chunks, median 229 tokens (p5 46, p95 389, max 453), 0 over 512 with header (max 474), 5.6% < 50 tokens, 97.8% have a heading. Identical to previous-year chunk: **14.8%** (MSFT 26%) vs 3% for plain packing | Full-corpus run not done yet. Heading is only the deepest level (e.g. TWLO "2022 compared to 2021" without its parent "Research and development"); a 2-level path is possible. FDS 2016–18 fall back to plain packing |
| 2026-11-24 | 2.2c Chunker (full corpus) | Ran on all 950 filings in **65 s**. Found + fixed 2 bugs that the small set didn't show: (1) a body paragraph opening with the same words as its risk title matched as a heading (100-char key collision) → `headings.lookup()` also checks total length (`n_chars` added to `data/headings/`, cache rebuilt); heading runs now also respect the cap; (2) a long risk title glued to the first sentence-split piece of a long paragraph overshot → that first piece is now split smaller. **Result: 170,187 chunks, 38.32M tokens**; median 227 tokens (p5 47, p95 388, max 452); embed text (with header) max 510 = exactly bge's 512 with its 2 special tokens, 0 truncated; 5.4% tiny (< 50); 99.3% have a heading. Item 1A 72.7K chunks (median 269), Item 7 56.0K, Item 1 37.8K, Item 7A 3.7K. Identical to previous-year chunk: **22.0%** (per company 5–44%) | Chunks are not stored yet → 2.3 writes them to Neo4j. Optional: 2-level heading path |
| 2026-11-24 | Restructure | `src/` → one installable package `sec10k` (src layout): `config.py` (all paths + `SECTIONS` + `load_manifest()`, loads `.env`), `llm/providers.py` (was `llm.py`), `ingest/` (loader, headings, chunker), `analysis/` (profile_corpus, compare_chunking). Page-noise regexes moved from the profiler into the loader, so the pipeline no longer imports an analysis script (analysis → ingest only). New `pyproject.toml` (no deps; pins stay in `requirements.txt`), installed with `pip install -e .`; `.gitignore` += `*.egg-info/`, `build/`. Run modules with `.venv/bin/python -m sec10k.<pkg>.<module>`. Verified from outside the project: all imports, headings (950 cached), loader report and chunker test set unchanged. README + CLAUDE.md layout updated | Older log rows mention the old `src/*.py` paths (left as history) |
| 2026-11-24 | 2.3 Structural graph | Installed + pinned `llama-index-graph-stores-neo4j==0.9.0` (no other pins changed). New `sec10k/graph/`: `store.py` (`get_graph_store()`, settings in `config.py`) and `structure.py`. `(Company)-[:FILED]->(Filing)-[:HAS_SECTION]->(Section)-[:HAS_CHUNK]->(Chunk)`. Company = LlamaIndex `EntityNode` (id = name; extracted mentions will resolve onto it), Chunk = `ChunkNode` (text + metadata: ticker, fiscal_year, section, seq, heading, n_tokens, content_hash, ...), Filing/Section = own Cypher with `__Node__` but **not** `__Entity__` (keeps them out of entity search). All MERGE by id (unique constraint) → idempotent; leftover chunks from a longer earlier chunking are deleted. Tested: re-run gives identical counts; a planted stale chunk is removed. **Full corpus in 68 s:** 130 companies, 950 filings, 3,783 sections, 170,187 chunks (207 MB text); edges match; 0 orphans; 0 count mismatches. Neo4j 2.1 GB RAM, 1.1 GB on disk. Fixed a bug in the check query (missing parentheses around `OR`) | No embeddings yet (2.4). Stale *sections* (a section that later gets skipped) aren't cleaned up, only stale chunks. Optional: `NEXT` edges between consecutive chunks for context expansion in Phase 6 |
| 2026-11-24 | 2.3 Structural graph | Installed + pinned `llama-index-graph-stores-neo4j==0.9.0` (no other pins changed). New `sec10k/graph/`: `store.py` (`get_graph_store()`, settings in `config.py`) and `structure.py`. `(Company)-[:FILED]->(Filing)-[:HAS_SECTION]->(Section)-[:HAS_CHUNK]->(Chunk)`. Company = LlamaIndex `EntityNode` (id = name; extracted mentions will resolve onto it), Chunk = `ChunkNode` (text + metadata: ticker, fiscal_year, section, seq, heading, n_tokens, content_hash, ...), Filing/Section = own Cypher with `__Node__` but **not** `__Entity__` (keeps them out of entity search). All MERGE by id (unique constraint) → idempotent; leftover chunks from a longer earlier chunking are deleted. Tested: re-run gives identical counts; a planted stale chunk is removed. **Full corpus in 68 s:** 130 companies, 950 filings, 3,783 sections, 170,187 chunks (207 MB text); edges match; 0 orphans; 0 count mismatches. Neo4j 2.1 GB RAM, 1.1 GB on disk. Fixed a bug in the check query (missing parentheses around `OR`) | No embeddings yet (2.4). Stale *sections* (a section that later gets skipped) aren't cleaned up, only stale chunks. Optional: `NEXT` edges between consecutive chunks for context expansion in Phase 6 |
| 2026-11-24 | 2.3 decision | **No `NEXT` edges between chunks.** Neighbour/"whole risk factor" context is already a lookup via `seq` + `heading` within a Section; extra edges would duplicate it and could leak into generic multi-hop traversals. Add later only if Phase 6 Cypher needs it | — |
| 2026-11-24 | 2.4 Embeddings (small set) | `EMBED_MODEL`/`EMBED_DIM` moved to `config.py` (chunker tokenizer and embedder must match). New `sec10k/embeddings.py` (`get_embed_model()`: bge-base, MPS, max 512, normalized; sentence-transformers adds bge's query prompt to queries only) and `sec10k/ingest/embed.py`: resumable job, re-embeds a chunk only if it has no vector, its text changed (`embedded_hash <> content_hash`) or the model changed; vector index `chunk_embedding` (768-d, cosine). `Chunk.embed_text()` → reusable `build_embed_text()`. **5 test companies: 8,466 chunks in 8 min, index ONLINE 100%.** Speed: profiled; Neo4j fetch is 0.06–0.13 s/batch, the bottleneck is the model on MPS: **~19–21 chunks/s, degrading to ~18 as the M3 heats up** → full corpus ≈ 2.3–2.5 h. fp16 test: 25 chunks/s (+30%), vectors cos ≥ 0.9995 vs fp32, top-10 neighbour overlap 9.96/10. Sanity search: relevant hits for 3 test questions; **same paragraph from several years fills the top-k** → de-duplicate in 2.5 | Decide fp32 vs fp16 for the full run. Heading-only changes (same text) don't trigger re-embedding |
| 2026-11-24 | 2.4 Embeddings (full) ✅ | Added `--limit N` to `embed.py` (stop after N chunks). Runs in the background under `caffeinate -i`, logs in `data/logs/embed_batch_NN.log`. Batch 1: 16,200 chunks in 19.4 min (avg 14/s: 28/s cool, throttles within minutes). Batch 3-min cool-down → batch 2: 16,200 in 16.8 min (avg 16/s). Paused at 24%, resumed with 7,200-chunk batches + 5-min cool-down: batches 3–11 took 5.0–5.9 min each (avg 20–24/s; smaller batches end before throttling bites). Bug: `caffeinate` wrapped only the embed, not the `sleep`, so the Mac slept mid-cool-down and stalled ~40 min; fixed by wrapping both: `caffeinate -i sh -c 'sleep 300 && …'`. Paused again at 62%, then batches 12–20 (4.5–5.2 min each, ~24/s). **Done 2026-11-25: 170,187 / 170,187 chunks embedded** (all 768-dim, embedded_hash = content_hash, model = bge-base-en-v1.5), vector index `chunk_embedding` ONLINE 100%. Total 20 batches, ~2.2 h of GPU time | Next: 2.5 vector-search Q&A with citations; de-duplicate near-identical year-over-year chunks in results |
| 2026-11-25 | 2.5a Vector search | Decisions: keep fp32 vectors; Phase 2 committed; Q&A answers via Groq; code in new `sec10k/retrieval/`. `retrieval/vector_search.py`: `search(question, k)` embeds the question (bge query prompt), fetches 5×k candidates from `chunk_embedding`, groups year-over-year repeats (same company, cosine ≥ 0.95, computed in Neo4j) into one `Hit` that lists all its years; `Hit.citation()` → "MSFT FY2016,…,2024, Item 1 Business". Neo4j's cosine score is (1+cos)/2 → converted back to plain cosine. Two bugs fixed: LlamaIndex `structured_query` silently drops lists ≥ 128 items (so vectors can't be fetched → compare in Neo4j); matching `:Chunk {id}` skips the unique index on `__Node__.id` (full scan → minutes) → use `:__Node__`. **Tested 3 questions:** 8 results = ~8 distinct passages (cyber risk: 8 companies; MSFT competitors: Windows paragraph FY2016–24 as one hit). Threshold check: same paragraph across FY2016–24 = 0.94–0.99 (grouped); FY2015 vs later = 0.85–0.87 and really is a different chunk (only 1 of 3 paragraphs shared). One query ≈ 8 s, mostly model load | Baseline weaknesses seen: no company/year filter (MSFT question still returned an OKTA hit; FY2015 over-represented); "semiconductor suppliers" returns generic sole-supplier risk text. Next: 2.5b Groq answer with citations |
| 2026-11-25 | 2.5b Q&A with citations ✅ | `retrieval/qa.py`: `answer(question, k=8, backend)` → `search()`, numbers passages [1]..[k] with company + `citation()` + heading, system prompt "use only the sources, cite [n] after every claim, say if the sources don't answer it"; answer cap 1,500 tokens (includes gpt-oss thinking); no LLM call when nothing is found. **Groq, 3 test questions: 2.5–3.8K tokens each** (in 2.2–3.4K, out 340–380) → fits 8K TPM; ~50 questions/day possible. MSFT competitors: correct and fully cited (Apple/Google for Windows, Amazon/Google/IBM/Oracle/… for Azure, Sony/Nintendo for Xbox). Cyber risks: 8 companies, each summary matches its source. **Semiconductor suppliers: 2 of 4 named companies unsupported** — AAPL and CDNS passages say "components", never "semiconductor/chip"; the model added it anyway. TDC (Intel chips) missed. Answers also rarely state the fiscal year as asked. **Phase 2 complete** | Citation caveat: a grouped hit shows the text of its best-scoring year but cites all years (MSFT [1] Xbox text is FY2023, cited as FY2016–24). Possible fixes: stricter prompt / show only the kept year in the source label. Phase 6 should beat the semiconductor question. Phase 2.5 commit pending |
| 2026-11-25 | 2.5 wrap-up | `benchmark.md` (in git): the 3 test questions with the exact Groq answers reviewed in 2.5b (copied, not re-run, since gpt-oss output varies per run), sources, token use and hand checks; later methods add a new "Run N" section. `COMEBACK_LATER.md`: 19 open items grouped by phase; **git-ignored** (local only) | Phase 2.5 commit pending |
