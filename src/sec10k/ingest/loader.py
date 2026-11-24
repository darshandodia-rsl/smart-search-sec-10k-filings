"""Phase 2.1: load filings from the manifest and clean each section.

Fixes found in Phase 1 (the corpus on disk stays read-only; we clean in memory):
  1. Drop a leading table of contents (PD 2019-23 Item 1 starts with one).
  2. Trim a section where a *later* item starts (DBX 2022 Item 1A ran on through Item 16)
     or where the financial statements start (ALRM 2015 Item 7A ran into Item 8 unlabeled).
  3. Drop page furniture ("- 53 -", "Table of Contents", "Apple Inc. | 2020 Form 10-K | 29",
     running headers "PART I" / "Item 1A" / the company name, "_____" separators,
     Oracle's per-page "Index to Financial Statements") and rejoin sentences a page break split.
  4. Drop number-only lines (flattened table cells: "$", "140,936"). They carry no meaning
     once the table layout is gone, and would only add noise to chunks and embeddings.
  5. Skip sections with almost no text left (EVTC 2023 Item 1 is a heading only).

Run from the project root:  .venv/bin/python -m sec10k.ingest.loader   (prints a cleaning report)
"""

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from sec10k.config import CORPUS, SECTIONS, load_manifest

# Page furniture the HTML-to-text step left behind: "- 53 -", "29", "Table of Contents",
# "Apple Inc. | 2020 Form 10-K | 29", "Leidos Holdings, Inc. Annual Report - 55".
PAGE_RE = re.compile(
    r"^\s*(?:-?\s*\d{1,3}\s*-?|tables? of contents|.{0,80}(?:form 10-k|annual report).{0,20}\d{1,3})\s*$",
    re.I,
)
# A line that is only numbers/currency/punctuation: a flattened table cell.
NUMLINE_RE = re.compile(r"^[\s$()%,.\-—–0-9]+$")

SECTION_TITLES = {
    "item_1": "Item 1 Business",
    "item_1A": "Item 1A Risk Factors",
    "item_7": "Item 7 MD&A",
    "item_7A": "Item 7A Market Risk",
}
# Order of 10-K items, used to decide whether a heading belongs to a *later* item.
ITEM_ORDER = ["1", "1a", "1b", "1c", "2", "3", "4", "5", "6", "7", "7a", "8",
              "9", "9a", "9b", "9c", "10", "11", "12", "13", "14", "15", "16"]
OWN_ITEM = {"item_1": "1", "item_1A": "1a", "item_7": "7", "item_7A": "7a"}

# An item heading line: "ITEM 2. PROPERTIES", "Item 1B.", "Item IA." (roman I for 1).
ITEM_HEADING_RE = re.compile(r"^\s*item\s*(\d{1,2}|i)\s*([a-c])?\s*[.:\-—–]", re.I)
FIN_STATEMENTS_RE = re.compile(r"^\s*index to (?:consolidated )?financial statements\s*$", re.I)
MAX_HEADING_LEN = 150   # longer lines are prose that happens to start with "Item 7 ..."
MIN_WORDS = 50          # below this a section is empty or heading-only (Phase 1 threshold)


@dataclass
class Section:
    cik: str
    ticker: str
    company: str
    fiscal_year: int
    period_of_report: str
    filing_date: str
    accession: str
    section: str        # item_1 | item_1A | item_7 | item_7A
    text: str           # cleaned; one paragraph per line


def item_key(line: str) -> str | None:
    """'ITEM 1A. Risk Factors' -> '1a'; None if the line is not a short item heading."""
    if len(line) > MAX_HEADING_LEN:
        return None
    m = ITEM_HEADING_RE.match(line)
    if not m:
        return None
    num = "1" if m.group(1).lower() == "i" else m.group(1)
    return num + (m.group(2) or "").lower()


def drop_leading_toc(lines: list[str]) -> tuple[list[str], int]:
    """A table of contents = a run of short lines at the top containing several item
    headings. Drop the run up to its last item heading (plus that heading's title line)."""
    last_heading, n_headings = -1, 0
    for i, ln in enumerate(lines):
        if len(ln) > MAX_HEADING_LEN:      # first real paragraph: the TOC (if any) is over
            break
        if item_key(ln):
            last_heading, n_headings = i, n_headings + 1
    if n_headings < 4:
        return lines, 0
    cut = last_heading + 1
    if cut < len(lines) and len(lines[cut]) <= MAX_HEADING_LEN and not item_key(lines[cut]):
        cut += 1                           # the title that follows "Item 16." on its own line
    return lines[cut:], cut


def trim_at_later_item(sec: str, lines: list[str]) -> tuple[list[str], str | None]:
    """Cut at the first heading of a later item or at the financial-statements index."""
    own = ITEM_ORDER.index(OWN_ITEM[sec])
    # Oracle repeats "Index to Financial Statements" as a link on every page: if the line
    # occurs more than once it is page furniture (dropped later), not where Item 8 starts.
    fin_is_boundary = sum(bool(FIN_STATEMENTS_RE.match(ln)) for ln in lines) == 1
    for i, ln in enumerate(lines[1:], start=1):   # line 0 is the section's own heading
        key = item_key(ln)
        if key in ITEM_ORDER and ITEM_ORDER.index(key) > own:
            return lines[:i], ln.strip()[:60]
        if fin_is_boundary and FIN_STATEMENTS_RE.match(ln):
            return lines[:i], ln.strip()[:60]
    return lines, None


# Running page headers: "PART I", bare "Item 1A", "Intuit Fiscal 2023 Form 10-K",
# "Veeva Systems Inc. | Form 10-K"; and separator lines like "_______".
RUNNING_HEADER_RE = re.compile(r"^(?:part\s+[ivx]+|item\s*\d{1,2}\s*[a-c]?\.?)$", re.I)
FORM_10K_RE = re.compile(r"form\s*10-?k", re.I)
NO_TEXT_RE = re.compile(r"^[^A-Za-z0-9]+$")
BULLETS = {"•", "●", "·", "▪", "◦", "■", "○", "-", "–", "—", "*"}


def is_page_furniture(line: str) -> bool:
    return bool(PAGE_RE.match(line) or FIN_STATEMENTS_RE.match(line)
                or RUNNING_HEADER_RE.match(line)
                or (FORM_10K_RE.search(line) and len(line.split()) <= 10)
                or (NO_TEXT_RE.match(line) and line not in BULLETS))


def company_header_lines(lines: list[str], company: str) -> set[str]:
    """Short lines that repeat the company's name 2+ times in a section: a page header
    ("Alphabet Inc.", "Blackbaud, Inc."). Matched on the name's first word."""
    first = re.sub(r"[^a-z0-9]", "", company.lower().split()[0])
    counts: dict[str, int] = {}
    for ln in lines:
        if len(ln.split()) <= 6 and re.sub(r"[^a-z0-9]", "", ln.lower().split()[0]) == first:
            counts[ln] = counts.get(ln, 0) + 1
    return {ln for ln, n in counts.items() if n >= 2}


def join_lone_bullets(lines: list[str]) -> tuple[list[str], int]:
    """Bullets laid out as a table come out as "•" on its own line: glue it to the next."""
    out, joined = [], 0
    for ln in lines:
        if out and out[-1] in BULLETS:
            out[-1] = f"{out[-1]}{ln}" if out[-1] in {"•", "●", "·", "▪", "◦", "■", "○"} else f"{out[-1]} {ln}"
            joined += 1
        else:
            out.append(ln)
    return out, joined


SENTENCE_END = tuple('.:;!?"”)')


def drop_page_lines(lines: list[str], extra: set[str] = frozenset()) -> tuple[list[str], int, int]:
    """Drop page furniture (plus any lines in `extra`). A page break often falls
    mid-sentence ("...may not be able to compete eff" / [footer] / "ectively ..."): rejoin
    the two halves when the text before the break has no end punctuation and the text
    after starts in lowercase. Line 0 (the section's own heading) is always kept."""
    out: list[str] = []
    dropped = rejoined = 0
    after_break = False
    for i, ln in enumerate(lines):
        if i > 0 and (is_page_furniture(ln) or ln in extra):
            dropped += 1
            after_break = True
            continue
        if after_break and out and not out[-1].endswith(SENTENCE_END) and ln[:1].islower():
            out[-1] = f"{out[-1]} {ln}"
            rejoined += 1
        else:
            out.append(ln)
        after_break = False
    return out, dropped, rejoined


def clean_section(sec: str, text: str, company: str) -> tuple[str, dict]:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    words_before = len(text.split())
    lines, toc_lines = drop_leading_toc(lines)
    lines, trimmed_at = trim_at_later_item(sec, lines)
    lines, page_lines, rejoined = drop_page_lines(lines, company_header_lines(lines, company))
    n = len(lines)
    lines = [ln for ln in lines if not NUMLINE_RE.match(ln)]
    number_lines = n - len(lines)
    lines, bullets_joined = join_lone_bullets(lines)
    cleaned = "\n".join(lines)
    stats = {
        "words_before": words_before,
        "words_after": len(cleaned.split()),
        "toc_lines_dropped": toc_lines,
        "trimmed_at": trimmed_at,
        "page_lines_dropped": page_lines,
        "sentences_rejoined": rejoined,
        "number_lines_dropped": number_lines,
        "bullets_joined": bullets_joined,
    }
    return cleaned, stats


def iter_clean_sections(tickers: list[str] | None = None,
                        stats_out: list[dict] | None = None) -> Iterator[Section]:
    """Yield cleaned sections for filings in the manifest (optionally only some tickers).
    Sections left with < MIN_WORDS words are skipped. Pass a list as stats_out to collect
    per-section cleaning stats (including skipped sections)."""
    manifest = load_manifest()
    if tickers:
        manifest = manifest[manifest["ticker"].isin(tickers)]
    for row in manifest.itertuples(index=False):
        filing = json.loads((CORPUS / row.json_path).read_text())
        for sec in SECTIONS:
            text, stats = clean_section(sec, filing.get(sec) or "", row.name)
            skipped = stats["words_after"] < MIN_WORDS
            if stats_out is not None:
                stats_out.append({"ticker": row.ticker, "fiscal_year": row.fiscal_year,
                                  "section": sec, "skipped": skipped, **stats})
            if skipped:
                continue
            yield Section(
                cik=row.cik, ticker=row.ticker, company=row.name,
                fiscal_year=int(row.fiscal_year), period_of_report=row.period_of_report,
                filing_date=row.filing_date, accession=row.accession_number,
                section=sec, text=text,
            )


def main() -> None:
    stats: list[dict] = []
    n = sum(1 for _ in iter_clean_sections(stats_out=stats))
    df = pd.DataFrame(stats)
    print(f"Sections in manifest: {len(df)}   kept: {n}   skipped (< {MIN_WORDS} words): {df['skipped'].sum()}")
    print(f"Words: {df['words_before'].sum() / 1e6:.2f}M before -> "
          f"{df.loc[~df['skipped'], 'words_after'].sum() / 1e6:.2f}M after\n")

    print("Sections touched by each fix:")
    print(f"  table of contents dropped: {(df['toc_lines_dropped'] > 0).sum()}")
    print(f"  trimmed at a later item:   {df['trimmed_at'].notna().sum()}")
    print(f"  page lines dropped:        {(df['page_lines_dropped'] > 0).sum()} "
          f"({df['page_lines_dropped'].sum()} lines, "
          f"{df['sentences_rejoined'].sum()} split sentences rejoined)")
    print(f"  number lines dropped:      {(df['number_lines_dropped'] > 0).sum()} "
          f"({df['number_lines_dropped'].sum()} lines)")
    print(f"  lone bullets joined:       {(df['bullets_joined'] > 0).sum()} "
          f"({df['bullets_joined'].sum()} lines)\n")

    cols = ["ticker", "fiscal_year", "section", "words_before", "words_after"]
    print("Trimmed sections:")
    print(df[df["trimmed_at"].notna()][cols + ["trimmed_at"]].to_string(index=False), "\n")
    print("TOC dropped:")
    print(df[df["toc_lines_dropped"] > 0][cols + ["toc_lines_dropped"]].to_string(index=False), "\n")
    print("Skipped:")
    print(df[df["skipped"]][cols].to_string(index=False))


if __name__ == "__main__":
    main()
