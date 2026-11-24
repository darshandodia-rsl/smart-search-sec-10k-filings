"""Phase 2.2 experiment: paragraph-aware vs semantic chunking, side by side.

Same filings, two LlamaIndex splitters:
  - paragraph: SentenceSplitter packs whole paragraphs (one per line) up to ~400 bge tokens;
               only paragraphs longer than that get split, at sentence breaks. No overlap.
  - semantic:  SemanticSplitterNodeParser embeds every sentence (with 1 neighbour each side)
               and starts a new chunk where similarity to the next sentence drops below the
               section's 95th-percentile distance (LlamaIndex default).
Measures: chunk count and size (bge tokens), chunks over bge's 512-token limit (silently
truncated when embedded), tiny chunks, run time, and year-over-year stability (share of
FY2023 chunks whose exact text also appears as a FY2022 chunk = work we could skip).

Run from the project root:  .venv/bin/python -m sec10k.analysis.compare_chunking
"""

import time

from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer

from sec10k.ingest.loader import iter_clean_sections

MODEL = "BAAI/bge-base-en-v1.5"
FILINGS = {("MSFT", 2022), ("MSFT", 2023), ("CRM", 2023)}
CHUNK_TOKENS = 400
BGE_LIMIT = 512

tok = AutoTokenizer.from_pretrained(MODEL)
n_tokens = lambda s: len(tok.encode(s, add_special_tokens=False))  # noqa: E731

embed = HuggingFaceEmbedding(model_name=MODEL, device="mps")
splitters = {
    "paragraph": SentenceSplitter(chunk_size=CHUNK_TOKENS, chunk_overlap=0, paragraph_separator="\n",
                                  tokenizer=lambda s: tok.encode(s, add_special_tokens=False)),
    "semantic": SemanticSplitterNodeParser(embed_model=embed, buffer_size=1,
                                           breakpoint_percentile_threshold=95),
}

sections = [s for s in iter_clean_sections(["MSFT", "CRM"]) if (s.ticker, s.fiscal_year) in FILINGS]
print(f"{len(sections)} sections, {sum(n_tokens(s.text) for s in sections):,} bge tokens\n")

results = {}
for name, splitter in splitters.items():
    t0 = time.time()
    chunks = {}  # (ticker, year, section) -> list of chunk texts
    for s in sections:
        nodes = splitter.get_nodes_from_documents([Document(text=s.text)])
        chunks[(s.ticker, s.fiscal_year, s.section)] = [n.get_content() for n in nodes]
    secs = time.time() - t0
    sizes = sorted(n_tokens(c) for cs in chunks.values() for c in cs)
    n = len(sizes)
    # Stability: MSFT FY2023 chunks that exist verbatim among MSFT FY2022 chunks (same section).
    same = total = 0
    for (t, y, sec), cs in chunks.items():
        if (t, y) == ("MSFT", 2023):
            prev = set(chunks.get(("MSFT", 2022, sec), []))
            same += sum(c in prev for c in cs)
            total += len(cs)
    results[name] = chunks
    print(f"== {name}: {n} chunks in {secs:.1f}s")
    print(f"   tokens  p5={sizes[n // 20]}  median={sizes[n // 2]}  p95={sizes[n * 19 // 20]}  max={sizes[-1]}")
    print(f"   > {BGE_LIMIT} tokens (truncated): {sum(x > BGE_LIMIT for x in sizes)} "
          f"({sum(max(0, x - BGE_LIMIT) for x in sizes):,} tokens lost)")
    print(f"   < 50 tokens (tiny): {sum(x < 50 for x in sizes)}")
    print(f"   MSFT FY2023 chunks identical to a FY2022 chunk: {same}/{total} ({same / total:.0%})\n")

# Show where each method puts its first few boundaries in MSFT FY2023 Risk Factors.
key = ("MSFT", 2023, "item_1A")
for name, chunks in results.items():
    print(f"--- {name}: first 6 chunks of MSFT FY2023 Item 1A (start ... end, tokens) ---")
    for c in chunks[key][:6]:
        flat = c.replace("\n", " / ")
        print(f"  [{n_tokens(c):>4}] {flat[:90]!r} ... {flat[-60:]!r}")
    print()
