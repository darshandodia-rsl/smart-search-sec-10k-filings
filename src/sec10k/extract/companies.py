"""Tier 2 extraction: find mentions of our 130 corpus companies in chunk text, without an LLM.

What it does: reads the hand-checked alias table (`company_aliases.csv`: the names each
company goes by, including former names) and matches every alias in a chunk's text, case-
sensitively and on whole words. An alias can carry two small regexes that cancel a match
when they appear right before or after it (e.g. "Box" but not "P.O. Box"). Each match
records which company it is, the exact text matched, its position, and whether it is the
filer itself ("self" mentions say nothing about relationships between companies).

Why it is needed: this is the cheapest extraction tier. It costs no tokens and runs over
the whole corpus in about a minute. Its matches become MENTIONS links from chunks to
Company nodes, and they help decide which chunks are worth sending to the LLM (a chunk that
names other companies is a likely source of competitor/supplier/partner facts).
Tickers are not matched in body text: most are also ordinary words (NOW, PATH, AI, U).

Run: .venv/bin/python -m sec10k.extract.companies [TICKER ...]   (report over all or some filings)
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from sec10k.config import COMPANY_ALIASES, load_companies


@dataclass(frozen=True)
class CompanyMention:
    """One match of a corpus company's name in a chunk."""

    ticker: str      # the company that was mentioned
    alias: str       # the alias that matched (e.g. "Facebook" for META)
    start: int       # character position of the match in the chunk text
    end: int
    is_self: bool    # True if the chunk belongs to the mentioned company's own filing


class CompanyMatcher:
    """Finds corpus-company names in text using the alias table.

    All aliases go into one regex (longest first, so "Zoom Video Communications" wins over
    "Zoom"), which keeps matching fast enough for 170K chunks. Per-alias exclusions are
    checked on the text around each match.
    """

    def __init__(self, aliases: pd.DataFrame | None = None):
        """Build the matcher from the alias table (read from `company_aliases.csv` by default)."""
        df = aliases if aliases is not None else load_aliases()
        known = set(load_companies().ticker)
        if unknown := set(df.ticker) - known:
            raise ValueError(f"alias table has tickers not in companies.csv: {sorted(unknown)}")
        if dupes := df.alias[df.alias.duplicated()].tolist():
            raise ValueError(f"alias used for more than one row: {dupes}")
        self.by_alias = {r.alias: r for r in df.itertuples()}
        self._before = {r.alias: re.compile(f"(?:{r.not_preceded_by})$")
                        for r in df.itertuples() if r.not_preceded_by}
        self._after = {r.alias: re.compile(f"^(?:{r.not_followed_by})")
                       for r in df.itertuples() if r.not_followed_by}
        alts = "|".join(re.escape(a) for a in sorted(self.by_alias, key=len, reverse=True))
        # Whole words only: no letter/digit (or '.', '&', '-') glued on either side,
        # so "Box" does not match inside "Dropbox" or "Box-like".
        self.pattern = re.compile(rf"(?<![\w.&-])(?:{alts})(?![\w&-])")

    def find(self, text: str, filer_ticker: str | None = None) -> list[CompanyMention]:
        """Return every corpus-company mention in `text`, in order of position.

        `filer_ticker` is the company whose filing the text comes from; mentions of it are
        flagged `is_self`.
        """
        out = []
        for m in self.pattern.finditer(text):
            alias = m.group(0)
            before = self._before.get(alias)
            if before and before.search(text[max(0, m.start() - 40):m.start()]):
                continue
            after = self._after.get(alias)
            if after and after.search(text[m.end():m.end() + 40]):
                continue
            ticker = self.by_alias[alias].ticker
            out.append(CompanyMention(ticker, alias, m.start(), m.end(), ticker == filer_ticker))
        return out


def load_aliases() -> pd.DataFrame:
    """Read the alias table; empty exclusion cells become empty strings."""
    return pd.read_csv(COMPANY_ALIASES, dtype=str, keep_default_na=False)


def report(tickers: list[str] | None = None) -> None:
    """Run the matcher over the chunks of all (or some) filings and print summary numbers.

    Shows how many chunks name another corpus company (the useful signal), which companies
    are named most, how often each alias fires, and companies that are never named. Used to
    check the alias table before the matches are written to the graph.
    """
    import time

    from sec10k.ingest.chunker import iter_chunks

    matcher = CompanyMatcher()
    t = time.time()
    n_chunks = n_with_other = n_with_2plus = 0
    by_company, by_alias, by_filer = Counter(), Counter(), Counter()
    for c in iter_chunks(tickers):
        n_chunks += 1
        mentions = matcher.find(c.text, c.ticker)
        others = {m.ticker for m in mentions if not m.is_self}
        for m in mentions:
            by_alias[m.alias] += 1
        by_company.update(others)
        if others:
            n_with_other += 1
            by_filer[c.ticker] += 1
        n_with_2plus += len(others) >= 2
    secs = time.time() - t

    print(f"{n_chunks:,} chunks in {secs:.0f} s")
    print(f"  naming >= 1 other corpus company: {n_with_other:,} ({n_with_other / n_chunks:.1%})")
    print(f"  naming >= 2 other corpus companies: {n_with_2plus:,} ({n_with_2plus / n_chunks:.1%})")
    print("\nMost-named companies (chunks, excluding self-mentions):")
    print("  " + ", ".join(f"{t} {n:,}" for t, n in by_company.most_common(20)))
    print("\nFilers whose chunks name other corpus companies most:")
    print("  " + ", ".join(f"{t} {n:,}" for t, n in by_filer.most_common(15)))
    print("\nMatches per alias (incl. self):")
    print("  " + ", ".join(f"{a} {n:,}" for a, n in by_alias.most_common()))
    never = sorted(set(load_companies().ticker) - set(by_company))
    print(f"\nNever named by another company ({len(never)}): {', '.join(never)}")


if __name__ == "__main__":
    import sys

    report(sys.argv[1:] or None)
