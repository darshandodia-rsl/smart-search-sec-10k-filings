"""Tier 2 extraction: find relationship cue phrases ("competitors include", "we rely on ...
suppliers", "acquired") in chunk text, without an LLM.

What it does: for each company-to-company relation in the schema (COMPETES_WITH,
HAS_SUPPLIER, HAS_CUSTOMER, PARTNERS_WITH, ACQUIRED) it holds a few regexes for phrases that
typically introduce that relationship. `find_cues()` returns every match, and marks whether a
name follows it in the same sentence ("we compete with Oracle and SAP" vs "we compete with
larger companies"), since only the first kind names the other side of the relationship.

Why it is needed: the cues do not say *who* the other company is (the LLM tier does that),
but they cheaply show which chunks state relationships at all. Together with the company
dictionary (`extract/companies.py`) and later local NLP, they decide which chunks are worth
LLM tokens.

Run: .venv/bin/python -m sec10k.extract.patterns [TICKER ...]   (report over all or some filings)
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

# --------------------------------------------------------------------------------------
# Cue phrases per relation. Case-insensitive. Kept short and readable on purpose: each one
# was checked by sampling its matches in the corpus.
# --------------------------------------------------------------------------------------
_CUES: dict[str, list[str]] = {
    "COMPETES_WITH": [
        r"\bcompetitors?,?\s+(?:include|including|such\s+as|are|is|like)\b",
        r"\bcompet(?:e|es|ing)\s+(?:\w+\s+){0,2}(?:with|against)\b",
        r"\bcompetition\s+(?:from|with)\b",
        r"\b(?:principal|primary|main|major|key|direct)\s+competitors?\b",
    ],
    "HAS_SUPPLIER": [
        r"\b(?:rel(?:y|ies|ied)|depend(?:s|ed)?)\s+(?:up)?on\s+(?:\w+\s+){0,6}"
        r"(?:suppliers?|manufacturers?|vendors?|foundr(?:y|ies)|providers?|data\s+centers?)\b",
        r"\b(?:suppliers?|manufacturers?)\s+(?:include|including|such\s+as)\b",
        r"\b(?:sole|single|limited)[-\s]sources?\b",
        r"\bmanufactured\s+(?:for\s+us\s+)?by\b",
        r"\bhosted\s+(?:\w+\s+){0,3}(?:on|by|with|in)\b",
    ],
    "HAS_CUSTOMER": [
        r"\b(?:largest|significant|major|top|key)\s+(?:\w+\s+)?customers?\b",
        r"\bcustomers?\s+(?:include|including|such\s+as)\b",
        # a revenue share only counts when a customer word comes first in the same sentence
        r"\b(?:customers?|clients?|resellers?|distributors?)\b[^.;]{0,120}?\baccount(?:ed|s)?\s+for\s+"
        r"(?:approximately\s+|about\s+|over\s+|more\s+than\s+)?"
        r"\d+(?:\.\d+)?\s?%\s+of\s+(?:our\s+)?(?:total\s+|net\s+)?(?:revenues?|sales)\b",
        r"\bcustomer\s+concentration\b",
    ],
    "PARTNERS_WITH": [
        r"\b(?:strategic\s+)?(?:partnerships?|alliances?|collaborations?|joint\s+ventures?)\s+with\b",
        r"\bpartner(?:s|ed|ing)?\s+with\b",
        r"\b(?:partners|resellers|distributors)\s+(?:include|including|such\s+as)\b",
    ],
    "ACQUIRED": [
        r"\b(?:we|the\s+company)\s+acquired\b",
        r"\b(?:completed|closed|announced)\s+(?:the|our|its)\s+acquisition\s+of\b",
        r"\bagree(?:d|ment)\s+to\s+acquire\b",
        r"\bacquisition\s+of\s+(?=[A-Z])",
    ],
}
CUES: dict[str, list[re.Pattern]] = {
    rel: [re.compile(p, re.IGNORECASE) for p in pats] for rel, pats in _CUES.items()
}
# "acquisition of" only counts when followed by a capitalized name; that lookahead must stay
# case-sensitive, so this one cue is compiled without IGNORECASE.
CUES["ACQUIRED"][-1] = re.compile(r"\b[Aa]cquisition\s+of\s+(?=[A-Z])")

# Capitalized words that start or fill sentences but are not names.
_NOT_NAMES = {
    "We", "Our", "Us", "The", "This", "These", "That", "Those", "In", "On", "For", "If", "As",
    "And", "Or", "Of", "To", "A", "An", "It", "Its", "Some", "Many", "Other", "Such", "Any",
    "Company", "Company's", "Item", "Part", "U.S.", "US", "United", "States", "Inc", "Inc.",
    "Corporation", "LLC", "Ltd", "However", "Also", "Additionally", "In addition",
    # common industry acronyms and generic capitalized terms that are not company names
    "OEM", "OEMs", "SaaS", "PaaS", "IaaS", "AI", "IT", "API", "APIs", "CRM", "ERP", "HCM",
    "SMS", "IoT", "ISV", "ISVs", "SMB", "SMBs", "GSI", "GSIs", "VAR", "VARs", "MSP", "MSPs",
    "Card", "Networks", "Cloud", "Platform", "Services", "Software", "Solutions", "Partners",
    # dates ("fiscal year ended March 31") are not names either
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December",
}
_NAME = re.compile(r"\b[A-Z][\w&.'’-]*")
_SENTENCE_END = re.compile(r"[.;:!?](?:\s|$)(?=\s*[A-Z•])|\n")


@dataclass(frozen=True)
class Cue:
    """One relationship cue phrase found in a chunk."""

    relation: str    # schema relation the phrase points to, e.g. "COMPETES_WITH"
    text: str        # the matched phrase
    start: int
    end: int
    named: bool      # True if a name follows in the same sentence (the other side is named)


def _name_follows(text: str, pos: int, window: int = 200) -> bool:
    """Return True if a capitalized, non-common word follows `pos` before the sentence ends."""
    tail = text[pos:pos + window]
    if end := _SENTENCE_END.search(tail):
        tail = tail[:end.start()]
    return any(m.group(0).rstrip(".'’") not in _NOT_NAMES for m in _NAME.finditer(tail))


def find_cues(text: str) -> list[Cue]:
    """Return every relationship cue phrase in `text`, ordered by position."""
    out = []
    for rel, patterns in CUES.items():
        for p in patterns:
            for m in p.finditer(text):
                out.append(Cue(rel, m.group(0), m.start(), m.end(), _name_follows(text, m.end())))
    return sorted(out, key=lambda c: c.start)


def report(tickers: list[str] | None = None) -> None:
    """Run cue finding (and the company dictionary) over all or some filings and print numbers.

    Shows, per relation, how many chunks have a cue and a *named* cue, and how cues overlap
    with corpus-company mentions. This tells us how many chunks state relationships at all,
    which is what the LLM filter needs to know.
    """
    import time

    from sec10k.extract.companies import CompanyMatcher
    from sec10k.ingest.chunker import iter_chunks

    matcher = CompanyMatcher()
    t = time.time()
    n = 0
    any_cue, named_cue, by_section = Counter(), Counter(), Counter()
    n_any = n_named = n_named_or_company = n_company = 0
    for c in iter_chunks(tickers):
        n += 1
        cues = find_cues(c.text)
        rels = {q.relation for q in cues}
        named = {q.relation for q in cues if q.named}
        any_cue.update(rels)
        named_cue.update(named)
        has_company = any(not m.is_self for m in matcher.find(c.text, c.ticker))
        n_any += bool(rels)
        n_named += bool(named)
        n_company += has_company
        if named or has_company:
            n_named_or_company += 1
            by_section[c.section] += 1
    print(f"{n:,} chunks in {time.time() - t:.0f} s\n")
    print(f"{'relation':15} {'any cue':>14} {'named cue':>14}")
    for rel in CUES:
        print(f"{rel:15} {any_cue[rel]:>7,} {any_cue[rel] / n:5.1%} {named_cue[rel]:>7,} {named_cue[rel] / n:5.1%}")
    print(f"\nchunks with any cue:            {n_any:,} ({n_any / n:.1%})")
    print(f"chunks with a named cue:        {n_named:,} ({n_named / n:.1%})")
    print(f"chunks naming a corpus company: {n_company:,} ({n_company / n:.1%})")
    print(f"named cue OR corpus company:    {n_named_or_company:,} ({n_named_or_company / n:.1%})")
    print("  by section: " + ", ".join(f"{s} {k:,}" for s, k in sorted(by_section.items())))


if __name__ == "__main__":
    import sys

    report(sys.argv[1:] or None)
