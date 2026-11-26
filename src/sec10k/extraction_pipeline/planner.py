"""Stage 1 of the extraction pipeline: decide which text of a wave needs the LLM.

What it does: for each company in a wave, walks its filings from the oldest fiscal year to
the newest and turns them into work items:
  - "extract" items: text the LLM has to read. Route A = an Item 1A risk title seen for the
    first time; route B = a chunk that names another company and whose exact text has not
    been planned for extraction before.
  - "reuse" items: text that is identical to text already planned for extraction (usually
    the same paragraph in an earlier year). These cost no LLM call: the graph writer copies
    the facts of the earlier twin, so the later fiscal year still gets its own fact edges.
Text that is neither a risk title nor flagged for route B produces no item.

Why oldest first: a repeat can only be skipped if its first copy is extracted. Walking every
company from its first filing forward guarantees that, and each piece of text goes to the
LLM exactly once, in the year it first appears.

Every item has a stable `item_id` built from the filing, the route and the chunk or title,
so planning the same wave twice gives the same items. The job store (next stage) relies on
this to never queue the same work twice.

Run (dry run: prints counts, makes no LLM calls, writes nothing):
  .venv/bin/python -m sec10k.extraction_pipeline.planner MSFT CRM [--first-years N]
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from dataclasses import dataclass

from sec10k.config import load_manifest
from sec10k.extract.companies import CompanyMatcher
from sec10k.extract.entity_cleaning import load_entities
from sec10k.extract.llm_extract import is_route_a_title, is_route_b
from sec10k.ingest.chunker import Chunk, iter_chunks


@dataclass(frozen=True)
class WorkItem:
    """One unit of extraction work: a risk title (route A) or a chunk (route B) of one filing.

    `kind` is "extract" (send to the LLM) or "reuse" (copy the facts of `source`, an earlier
    item with identical text). `source` is the item_id of that earlier item, None for
    extract items. `tokens` is the text size in embedding-model tokens, used for estimates.
    """

    item_id: str
    kind: str
    route: str
    ticker: str
    fiscal_year: int
    accession: str
    chunk_id: str
    title: str | None
    source: str | None
    tokens: int


def make_item_id(accession: str, route: str, key: str) -> str:
    """Stable ID of a work item: the same filing, route and chunk/title always give the same ID."""
    return hashlib.sha1(f"{accession}|{route}|{key}".encode()).hexdigest()[:16]


def wave_filings(tickers: list[str], first_years: int | None = None) -> dict[str, list[int]]:
    """Return, per ticker, the fiscal years to plan, oldest first, taken from the manifest.

    With `first_years`, only each company's first N filings are taken (used for small test
    waves); a wave never starts in the middle of a company's history.
    """
    m = load_manifest()
    out = {}
    for t in tickers:
        years = sorted(m.loc[m.ticker == t, "fiscal_year"].tolist())
        if not years:
            raise ValueError(f"{t} is not in the manifest")
        out[t] = years[:first_years] if first_years else years
    return out


def plan_company(ticker: str, years: list[int], matcher: CompanyMatcher) -> list[WorkItem]:
    """Plan one company's filings for the given fiscal years (which must start at its first filing).

    Reads the company's chunks once, then walks the years in order. Two maps remember what is
    already planned for extraction: risk titles (route A) and chunk fingerprints (route B).
    A title or chunk found there becomes a "reuse" item pointing to the first copy.
    """
    by_year: dict[int, list[Chunk]] = defaultdict(list)
    for c in iter_chunks([ticker]):
        if c.fiscal_year in years:
            by_year[c.fiscal_year].append(c)

    items: list[WorkItem] = []
    first_title: dict[str, str] = {}   # risk title -> item_id of its extract item
    first_text: dict[str, str] = {}    # content_hash -> item_id of its extract item
    for fy in years:
        chunks = by_year.get(fy, [])
        if not chunks:
            continue
        accession = chunks[0].accession
        ents = load_entities(accession)
        titles_this_filing: set[str] = set()
        for c in chunks:
            # Route A: one item per distinct risk title of the filing, at the chunk where it first appears.
            if is_route_a_title(c) and c.heading not in titles_this_filing:
                titles_this_filing.add(c.heading)
                iid = make_item_id(accession, "A", c.heading)
                src = first_title.get(c.heading)
                if src is None:
                    first_title[c.heading] = iid
                items.append(WorkItem(iid, "reuse" if src else "extract", "A", ticker, fy, accession,
                                      c.chunk_id, c.heading, src, len(c.heading) // 4))
            # Route B: flagged chunks; identical text already planned is reused.
            if is_route_b(c, ents.get(c.chunk_id, []), matcher, ticker):
                iid = make_item_id(accession, "B", c.chunk_id)
                src = first_text.get(c.content_hash)
                if src is None:
                    first_text[c.content_hash] = iid
                items.append(WorkItem(iid, "reuse" if src else "extract", "B", ticker, fy, accession,
                                      c.chunk_id, None, src, c.n_tokens))
    return items


def plan_wave(tickers: list[str], first_years: int | None = None) -> list[WorkItem]:
    """Plan a whole wave: every given company from its first filing forward."""
    matcher = CompanyMatcher()
    items: list[WorkItem] = []
    for ticker, years in wave_filings(tickers, first_years).items():
        items.extend(plan_company(ticker, years, matcher))
    return items


def summarize(items: list[WorkItem]) -> None:
    """Print, per company and in total, how many items need the LLM and how many are reused."""
    rows: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    filings: dict[str, set] = defaultdict(set)
    for it in items:
        for key in (it.ticker, "TOTAL"):
            rows[key][f"{it.route}_{it.kind}"] += 1
            if it.kind == "extract":
                rows[key][f"{it.route}_tokens"] += it.tokens
            filings[key].add(it.accession)
    print(f"{'':8}{'filings':>8}{'A new':>8}{'A reuse':>9}{'B new':>8}{'B reuse':>9}{'B new tokens':>14}")
    for key in sorted(k for k in rows if k != "TOTAL") + ["TOTAL"]:
        r = rows[key]
        print(f"{key:8}{len(filings[key]):>8}{r['A_extract']:>8}{r['A_reuse']:>9}"
              f"{r['B_extract']:>8}{r['B_reuse']:>9}{r['B_tokens']:>14,}")
    t = rows["TOTAL"]
    b_all = t["B_extract"] + t["B_reuse"]
    if b_all:
        print(f"Route B: {t['B_reuse'] / b_all:.0%} of flagged chunks are reused (no LLM call).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dry-run the planner for a wave of companies.")
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--first-years", type=int, help="only each company's first N filings")
    a = ap.parse_args()
    summarize(plan_wave(a.tickers, a.first_years))
