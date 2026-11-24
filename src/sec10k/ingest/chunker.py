"""Phase 2.2b: structure-aware paragraph chunking.

Rules (sizes in bge tokens, the embedding model's tokenizer):
  1. A paragraph (one line of cleaned text) is never split, unless it alone exceeds
     MAX_TOKENS; then it is split at sentence breaks.
  2. Paragraphs are packed into a chunk up to MAX_TOKENS.
  3. A heading line (fully bold/italic/underlined in the HTML) or a run-in title paragraph
     ("Competition. We face ...") always starts a new chunk. Consecutive headings stay
     together ("RISKS RELATING TO ..." + the first risk title). So a chunk never spans two
     risk factors / subsections, and a heading is never left at the end of a chunk.
  4. Each chunk remembers the nearest heading. It goes into the embedding header, so the
     2nd half of a long risk factor still says which risk it belongs to.

IDs: chunk_id = "{accession}:{section}:{seq:04d}" is unique per filing position (stable as
long as the input and these rules don't change). content_hash = sha1 of the text: equal
across years when the text is unchanged (used for "seen before?" checks, never as an ID,
or identical chunks of two filings would be merged into one node).

Run from the project root:  .venv/bin/python -m sec10k.ingest.chunker [TICKER ...]
(no tickers = full corpus)
"""

import hashlib
import sys
from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd
from llama_index.core.node_parser import SentenceSplitter
from transformers import AutoTokenizer

from sec10k.config import EMBED_MODEL, load_manifest
from sec10k.ingest.headings import load_headings, lookup
from sec10k.ingest.loader import SECTION_TITLES, Section, iter_clean_sections

MAX_TOKENS = 400          # chunk text; header adds ~20-60, still under bge's 512 limit
BGE_LIMIT = 512
MAX_HEADING_WORDS = 30    # long risk titles are cut in the header to keep it short

_tok = AutoTokenizer.from_pretrained(EMBED_MODEL)


def count_tokens(texts: list[str]) -> list[int]:
    return [len(ids) for ids in _tok(texts, add_special_tokens=False)["input_ids"]]


_long_par_splitter = SentenceSplitter(
    chunk_size=MAX_TOKENS, chunk_overlap=0,
    tokenizer=lambda s: _tok.encode(s, add_special_tokens=False),
)


@dataclass
class Chunk:
    chunk_id: str
    cik: str
    ticker: str
    company: str
    fiscal_year: int
    filing_date: str
    accession: str
    section: str
    seq: int
    heading: str | None     # nearest heading / risk title above (or at the start of) the chunk
    text: str
    n_tokens: int
    content_hash: str

    def embed_text(self) -> str:
        return build_embed_text(self.company, self.ticker, self.fiscal_year, self.section,
                                self.heading, self.text)


def build_embed_text(company: str, ticker: str, fiscal_year: int, section: str,
                     heading: str | None, text: str) -> str:
    """Text sent to the embedding model: a context header + the chunk text, e.g.
    "ORACLE CORP (ORCL) 10-K FY2020 | Item 1A Risk Factors | <heading>\n<text>".
    The heading is left out when the text already starts with it."""
    parts = [f"{company} ({ticker}) 10-K FY{fiscal_year}", SECTION_TITLES[section]]
    if heading and not text.startswith(heading):
        words = heading.split()
        parts.append(" ".join(words[:MAX_HEADING_WORDS]) + (" ..." if len(words) > MAX_HEADING_WORDS else ""))
    return " | ".join(parts) + "\n" + text


def chunk_section(sec: Section, headings: dict[str, dict]) -> list[Chunk]:
    lines = sec.text.split("\n")
    sizes = count_tokens(lines)
    blocks = [lookup(headings, ln) for ln in lines]
    kinds = [b["kind"] if b else None for b in blocks]                     # full | lead | None

    groups: list[tuple[str | None, list[str]]] = []   # (heading, paragraphs) per chunk
    cur: list[str] = []
    cur_tokens = 0
    cur_heading: str | None = None   # heading recorded for the chunk being built
    heading: str | None = None       # most recent heading seen in the section
    only_headings = False            # current chunk holds only heading lines so far

    def flush():
        nonlocal cur, cur_tokens, only_headings
        if cur:
            groups.append((cur_heading, cur))
        cur, cur_tokens, only_headings = [], 0, False

    for line, size, kind, block in zip(lines, sizes, kinds, blocks):
        headings_pending = only_headings           # cur holds only heading lines right now
        if kind:                                   # heading line or run-in title paragraph
            if not only_headings or cur_tokens + size > MAX_TOKENS:
                flush()                            # rule 3: start a new chunk (or chunk full)
            # run-in title: keep the original-case emphasized prefix ("Competition.")
            heading = line if kind == "full" else line[:len(block["lead"])].strip()
            cur_heading = heading                  # deepest heading wins (risk title, not group)
            only_headings = kind == "full"
        else:
            only_headings = False

        if size > MAX_TOKENS:                      # rule 1: one huge paragraph -> sentences
            pieces = _long_par_splitter.split_text(line)
            if cur and headings_pending and not kind:
                # keep the pending headings with the first piece; make that piece smaller
                # so heading + piece still fit in MAX_TOKENS
                room = max(MAX_TOKENS - cur_tokens, 100)
                small = SentenceSplitter(chunk_size=room, chunk_overlap=0,
                                         tokenizer=_long_par_splitter._tokenizer).split_text(line)
                rest = _long_par_splitter.split_text(" ".join(small[1:])) if len(small) > 1 else []
                pieces = ["\n".join(cur + [small[0]])] + rest
                cur, cur_tokens = [], 0
            flush()
            groups.extend((heading, [p]) for p in pieces)
            continue
        if cur and cur_tokens + size > MAX_TOKENS:  # rule 2: chunk full
            flush()
        if not cur:
            cur_heading = heading
        cur.append(line)
        cur_tokens += size
    flush()

    chunks = []
    for seq, (heading, pars) in enumerate(groups):
        text = "\n".join(pars)
        chunks.append(Chunk(
            chunk_id=f"{sec.accession}:{sec.section}:{seq:04d}",
            cik=sec.cik, ticker=sec.ticker, company=sec.company, fiscal_year=sec.fiscal_year,
            filing_date=sec.filing_date, accession=sec.accession, section=sec.section, seq=seq,
            heading=heading, text=text, n_tokens=0,
            content_hash=hashlib.sha1(text.encode()).hexdigest()[:16],
        ))
    for c, n in zip(chunks, count_tokens([c.text for c in chunks])):
        c.n_tokens = n
    return chunks


def iter_chunks(tickers: list[str] | None = None) -> Iterator[Chunk]:
    json_path = {r.accession_number: r.json_path for r in load_manifest().itertuples()}
    cache: dict[str, dict] = {}
    for sec in iter_clean_sections(tickers):
        if sec.accession not in cache:
            cache.clear()                # sections of one filing arrive together
            cache[sec.accession] = load_headings(json_path[sec.accession])
        yield from chunk_section(sec, cache[sec.accession])


def report(chunks: list[Chunk]) -> None:
    df = pd.DataFrame([{**c.__dict__, "text": None} for c in chunks])
    df["embed_tokens"] = count_tokens([c.embed_text() for c in chunks])
    q = df["n_tokens"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).astype(int).to_dict()
    print(f"{len(df):,} chunks from {df['accession'].nunique()} filings, "
          f"{df['n_tokens'].sum() / 1e6:.2f}M text tokens")
    print(f"text tokens: p5={q[0.05]} p25={q[0.25]} median={q[0.5]} p75={q[0.75]} "
          f"p95={q[0.95]} max={df['n_tokens'].max()}")
    print(f"embed text (with header) max={df['embed_tokens'].max()}, over {BGE_LIMIT}: "
          f"{(df['embed_tokens'] > BGE_LIMIT).sum()}")
    print(f"tiny (< 50 tokens): {(df['n_tokens'] < 50).sum()} "
          f"({(df['n_tokens'] < 50).mean():.1%})   with a heading: {df['heading'].notna().mean():.1%}\n")
    print("Per section:")
    print(df.groupby("section")["n_tokens"].agg(chunks="count", median="median").astype(int), "\n")

    # Year-over-year: share of a filing's chunks whose exact text was in the previous year's
    # same section (same company). Plain 400-token packing scored 3% on MSFT FY2023.
    prev = df.set_index(["ticker", "section", "fiscal_year"])["content_hash"]
    groups = {k: set(v) for k, v in prev.groupby(level=[0, 1, 2])}
    same = df.apply(lambda r: r["content_hash"] in groups.get((r["ticker"], r["section"], r["fiscal_year"] - 1), ()), axis=1)
    has_prev = df.apply(lambda r: (r["ticker"], r["section"], r["fiscal_year"] - 1) in groups, axis=1)
    print(f"Chunks identical to a previous-year chunk: {same[has_prev].mean():.1%}")
    print((same[has_prev].groupby(df["ticker"]).mean() * 100).round(1).to_string(), "\n")


def show(chunks: list[Chunk], ticker: str, year: int, section: str, n: int = 6) -> None:
    print(f"--- {ticker} FY{year} {section}: first {n} chunks ---")
    for c in [c for c in chunks if (c.ticker, c.fiscal_year, c.section) == (ticker, year, section)][:n]:
        flat = c.text.replace("\n", " / ")
        print(f"[{c.n_tokens:>3}] heading={c.heading[:60] if c.heading else None!r}")
        print(f"      {flat[:110]!r} ... {flat[-50:]!r}")
    print()


def main() -> None:
    tickers = sys.argv[1:] or None
    chunks = list(iter_chunks(tickers))
    report(chunks)
    show(chunks, "MSFT", 2023, "item_1A")


if __name__ == "__main__":
    main()
