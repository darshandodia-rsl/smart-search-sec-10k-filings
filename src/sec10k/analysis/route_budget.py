"""Measure the LLM budget of extraction routes A and B from the real tier-2 and tier-3 results.

What it does: goes through every chunk once and decides
  - is it new? (the same company did not have the exact same chunk in an earlier filing;
    repeated chunks reuse a cached LLM result, so they cost nothing),
  - is it flagged for route B? (tier 2: a corpus-company name or a named relationship cue;
    tier 3: a cleaned GLiNER company other than the filer),
  - how many tokens does it have (o200k_harmony, gpt-oss's tokenizer)?
It also counts route A: the distinct Item 1A risk titles per company. It prints the tokens
per route, the split of route B by section and by which tier flagged it, and the days of LLM
work, using the Phase 1 per-day capacity (`profile_corpus.content_per_day()`).

Why it is needed: the routing decision (A + B + D, C decided later) was made with a route-B
size projected from a 300-chunk sample. This replaces that guess with a measured number, so
the extraction budget for Phase 3/4 is known before any LLM call is made.

Needs the cleaned entity files (`python -m sec10k.extract.entity_cleaning clean`).
Run: .venv/bin/python -m sec10k.analysis.route_budget
"""

from __future__ import annotations

import json
import time
from collections import Counter

from sec10k.analysis.profile_corpus import ENC, content_per_day
from sec10k.extract.companies import CompanyMatcher
from sec10k.extract.entity_cleaning import _clean_path
from sec10k.extract.patterns import find_cues
from sec10k.ingest.chunker import iter_chunks


def main() -> None:
    """Count route-A and route-B tokens over the whole corpus and print the budget."""
    t0 = time.time()
    matcher = CompanyMatcher()
    seen: set[tuple[str, str]] = set()          # (ticker, content_hash) already met
    titles: set[tuple[str, str]] = set()        # (ticker, risk title) for route A
    flag3: dict[str, bool] = {}                 # chunk_id -> tier-3 flag of the current filing
    current = None
    tok = Counter()                             # token totals by bucket
    n = Counter()                               # chunk counts by bucket
    for c in iter_chunks():
        if c.accession != current:
            current = c.accession
            flag3 = {x["chunk_id"]: x["names_other_company"]
                     for x in json.loads(_clean_path(c.accession).read_text())["chunks"]}
        t = len(ENC.encode(c.text))
        tok["all"] += t
        n["all"] += 1
        if c.section == "item_1A" and c.heading:
            titles.add((c.ticker, c.heading))
        key = (c.ticker, c.content_hash)
        if key in seen:
            continue
        seen.add(key)
        tok["new"] += t
        n["new"] += 1
        t2 = (any(q.named for q in find_cues(c.text))
              or any(not m.is_self for m in matcher.find(c.text, c.ticker)))
        t3 = flag3.get(c.chunk_id, False)
        if not (t2 or t3):
            continue
        source = "both" if t2 and t3 else ("tier2 only" if t2 else "tier3 only")
        for b in ("B", f"B:{c.section}", f"B:{source}"):
            tok[b] += t
            n[b] += 1

    title_tok = sum(len(ENC.encode(h)) for _, h in titles)
    speed = content_per_day()
    both = speed["groq"] + speed["claude"]
    m = 1e6

    def days(x: float) -> str:
        return f"{x / both:5.1f} days both | {x / speed['groq']:5.0f} days Groq only"

    print(f"{n['all']:,} chunks, {tok['all'] / m:.1f}M tokens; new (not repeated): "
          f"{n['new']:,} chunks, {tok['new'] / m:.1f}M tokens   [{time.time() - t0:.0f} s]\n")
    print(f"Route A  risk titles   {len(titles):>7,} titles   {title_tok / m:5.2f}M tokens   {days(title_tok)}")
    print(f"Route B  flagged new   {n['B']:>7,} chunks   {tok['B'] / m:5.2f}M tokens   {days(tok['B'])}")
    print(f"A + B                                     {(title_tok + tok['B']) / m:5.2f}M tokens   "
          f"{days(title_tok + tok['B'])}")
    print(f"(every new chunk: {tok['new'] / m:.1f}M tokens, {days(tok['new'])})\n")
    print("Route B by section:")
    for s in ("item_1", "item_1A", "item_7", "item_7A"):
        print(f"  {s:8} {n['B:' + s]:>7,} chunks  {tok['B:' + s] / m:5.2f}M tokens")
    print("Route B by which tier flagged the chunk:")
    for s in ("tier2 only", "both", "tier3 only"):
        print(f"  {s:11} {n['B:' + s]:>7,} chunks  {tok['B:' + s] / m:5.2f}M tokens")
    print(f"\nCapacity used (Phase 1 estimate, not yet measured): {both / 1e3:,.0f}K text tokens/day with "
          f"both backends, {speed['groq'] / 1e3:,.0f}K Groq only")


if __name__ == "__main__":
    main()
