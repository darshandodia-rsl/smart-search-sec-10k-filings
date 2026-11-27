"""Cuts each cleaned section into chunks: small, self-contained pieces of text.

What it does: splits every section into chunks of at most 400 tokens (a token is a small
piece of a word, counted with the embedding model's own tokenizer). It follows the
filing's structure instead of cutting blindly:
  1. A paragraph (one line of cleaned text) is kept whole. Only a paragraph that is longer
     than 400 tokens by itself is split, at sentence boundaries.
  2. Paragraphs are packed together into a chunk until it is full.
  3. A heading (a line that is fully bold, italic or underlined in the original HTML) or a
     run-in title ("Competition. We face ...") always starts a new chunk. Consecutive
     headings stay together, and a heading is never left alone at the end of a chunk. So a
     chunk never mixes two risk factors or two subsections.
  4. Each chunk remembers the nearest heading above it. That heading goes into the
     embedding header, so the second half of a long risk factor still says which risk it
     belongs to.

Why it is needed: chunks are the unit everything else works on. They are the nodes in the
graph, the pieces that get embedded for search, the pieces that get cited in answers, and
the pieces the entity extraction reads. Chunks that follow the document's structure give
cleaner search results and cleaner extraction.

IDs: `chunk_id` = "{accession}:{section}:{seq}" identifies a chunk by its position in a
filing. `content_hash` is a fingerprint of the text: it is the same across years when the
text is unchanged, so it is used to check "have we seen this text before?" and never as an
ID (two filings with identical text would otherwise collapse into one node).

Run from the project root:  .venv/bin/python -m sec10k.ingest.chunker [TICKER ...]
(no tickers = full corpus; prints a size report)
"""

import hashlib
import sys
from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd
from llama_index.core.node_parser import SentenceSplitter
from transformers import AutoTokenizer

from sec10k.config import BGE_MAX_TOKENS, EMBED_MODEL, load_manifest
from sec10k.ingest.headings import load_headings, lookup
from sec10k.ingest.loader import SECTION_TITLES, Section, iter_clean_sections

MAX_TOKENS = 400          # most tokens of text per chunk; the header adds ~20-60 more, staying under 512
MAX_HEADING_WORDS = 30    # headings longer than this are cut short when put in the header

_tok = AutoTokenizer.from_pretrained(EMBED_MODEL)


def count_tokens(texts: list[str]) -> list[int]:
    """Count the embedding-model tokens in each text, so chunk sizes are measured the way the model sees them."""
    return [len(ids) for ids in _tok(texts, add_special_tokens=False)["input_ids"]]


# Splits one very long paragraph into pieces of at most MAX_TOKENS, cutting at sentence ends.
_long_par_splitter = SentenceSplitter(
    chunk_size=MAX_TOKENS, chunk_overlap=0,
    tokenizer=lambda s: _tok.encode(s, add_special_tokens=False),
)


@dataclass
class Chunk:
    """One chunk of a section, with the details needed to store it in the graph and cite it.

    Holds the text, where it came from (company, year, section, position), its nearest
    heading, its size in tokens and a fingerprint of the text. The graph writer and the
    embedder both read these chunks.
    """

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
        """The text to embed for this chunk: a context header plus the chunk text."""
        return build_embed_text(self.company, self.ticker, self.fiscal_year, self.section,
                                self.heading, self.text)


def build_embed_text(company: str, ticker: str, fiscal_year: int, section: str,
                     heading: str | None, text: str) -> str:
    """Build the text that is sent to the embedding model: a context header plus the chunk text.

    Example: "ORACLE CORP (ORCL) 10-K FY2020 | Item 1A Risk Factors | <heading>\n<text>".
    The header tells the model which company, year and section a chunk is from, so similar
    wording in different filings still gets distinguishable vectors. The heading is left out
    when the chunk text already starts with it. It is a separate function (not only a Chunk
    method) so the embedder can rebuild the same text from what is stored in Neo4j.
    """
    parts = [f"{company} ({ticker}) 10-K FY{fiscal_year}", SECTION_TITLES[section]]
    if heading and not text.startswith(heading):
        words = heading.split()
        parts.append(" ".join(words[:MAX_HEADING_WORDS]) + (" ..." if len(words) > MAX_HEADING_WORDS else ""))
    return " | ".join(parts) + "\n" + text


def chunk_section(sec: Section, headings: dict[str, dict]) -> list[Chunk]:
    """Cut one cleaned section into chunks, following the rules in the module docstring.

    `headings` says which lines of the section are headings or run-in titles (from
    `headings.py`). This is the core of the chunker; everything else feeds it or reports on it.
    """
    lines = sec.text.split("\n")
    sizes = count_tokens(lines)
    blocks = [lookup(headings, ln) for ln in lines]
    kinds = [b["kind"] if b else None for b in blocks]                     # full | lead | None

    groups: list[tuple[str | None, list[str]]] = []   # one entry per chunk: (heading, its paragraphs)
    cur: list[str] = []
    cur_tokens = 0
    cur_heading: str | None = None   # heading stored on the chunk being built
    heading: str | None = None       # most recent heading seen in the section
    only_headings = False            # True while the chunk being built has only heading lines

    def flush():
        """Close the chunk being built (if any) and start an empty one."""
        nonlocal cur, cur_tokens, only_headings
        if cur:
            groups.append((cur_heading, cur))
        cur, cur_tokens, only_headings = [], 0, False

    for line, size, kind, block in zip(lines, sizes, kinds, blocks):
        headings_pending = only_headings           # cur holds only heading lines right now
        if kind:                                   # heading line or run-in title paragraph
            if not only_headings or cur_tokens + size > MAX_TOKENS:
                flush()                            # headings start a new chunk (or the chunk is full)
            # run-in title: keep the original-case emphasized prefix ("Competition.")
            heading = line if kind == "full" else line[:len(block["lead"])].strip()
            cur_heading = heading                  # deepest heading wins (risk title, not group)
            only_headings = kind == "full"
        else:
            only_headings = False

        if size > MAX_TOKENS:                      # a paragraph too long for one chunk: split at sentences
            pieces = _long_par_splitter.split_text(line)
            if cur and headings_pending and not kind:
                # keep the headings waiting in the current chunk together with the first piece,
                # and make that piece smaller so headings + piece still fit in MAX_TOKENS
                room = max(MAX_TOKENS - cur_tokens, 100)
                small = SentenceSplitter(chunk_size=room, chunk_overlap=0,
                                         tokenizer=_long_par_splitter._tokenizer).split_text(line)
                rest = _long_par_splitter.split_text(" ".join(small[1:])) if len(small) > 1 else []
                pieces = ["\n".join(cur + [small[0]])] + rest
                cur, cur_tokens = [], 0
            flush()
            groups.extend((heading, [p]) for p in pieces)
            continue
        if cur and cur_tokens + size > MAX_TOKENS:  # the paragraph does not fit: close this chunk
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
    """Yield the chunks of every filing in the manifest (or only the given tickers).

    Loads and cleans each section, looks up its headings, and chunks it. The graph writer
    reads chunks from here, so the graph always reflects the current chunking rules.
    """
    json_path = {r.accession_number: r.json_path for r in load_manifest().itertuples()}
    cache: dict[str, dict] = {}
    for sec in iter_clean_sections(tickers):
        if sec.accession not in cache:
            cache.clear()                # keep only the current filing's headings in memory
            cache[sec.accession] = load_headings(json_path[sec.accession])
        yield from chunk_section(sec, cache[sec.accession])


def report(chunks: list[Chunk]) -> None:
    """Print size statistics for a list of chunks (a quality check on the chunking).

    Shows chunk counts and token sizes, how many are tiny or over the model limit, how many
    have a heading, and how many chunks repeat exactly from the previous year.
    """
    df = pd.DataFrame([{**c.__dict__, "text": None} for c in chunks])
    df["embed_tokens"] = count_tokens([c.embed_text() for c in chunks])
    q = df["n_tokens"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).astype(int).to_dict()
    print(f"{len(df):,} chunks from {df['accession'].nunique()} filings, "
          f"{df['n_tokens'].sum() / 1e6:.2f}M text tokens")
    print(f"text tokens: p5={q[0.05]} p25={q[0.25]} median={q[0.5]} p75={q[0.75]} "
          f"p95={q[0.95]} max={df['n_tokens'].max()}")
    print(f"embed text (with header) max={df['embed_tokens'].max()}, over {BGE_MAX_TOKENS}: "
          f"{(df['embed_tokens'] > BGE_MAX_TOKENS).sum()}")
    print(f"tiny (< 50 tokens): {(df['n_tokens'] < 50).sum()} "
          f"({(df['n_tokens'] < 50).mean():.1%})   with a heading: {df['heading'].notna().mean():.1%}\n")
    print("Per section:")
    print(df.groupby("section")["n_tokens"].agg(chunks="count", median="median").astype(int), "\n")

    # Year-over-year: share of a filing's chunks whose exact text also appears in the same
    # section of the same company's previous-year filing.
    prev = df.set_index(["ticker", "section", "fiscal_year"])["content_hash"]
    groups = {k: set(v) for k, v in prev.groupby(level=[0, 1, 2])}
    same = df.apply(lambda r: r["content_hash"] in groups.get((r["ticker"], r["section"], r["fiscal_year"] - 1), ()), axis=1)
    has_prev = df.apply(lambda r: (r["ticker"], r["section"], r["fiscal_year"] - 1) in groups, axis=1)
    print(f"Chunks identical to a previous-year chunk: {same[has_prev].mean():.1%}")
    print((same[has_prev].groupby(df["ticker"]).mean() * 100).round(1).to_string(), "\n")


def show(chunks: list[Chunk], ticker: str, year: int, section: str, n: int = 6) -> None:
    """Print the first `n` chunks of one section, to check by eye that the cuts look right."""
    print(f"--- {ticker} FY{year} {section}: first {n} chunks ---")
    for c in [c for c in chunks if (c.ticker, c.fiscal_year, c.section) == (ticker, year, section)][:n]:
        flat = c.text.replace("\n", " / ")
        print(f"[{c.n_tokens:>3}] heading={c.heading[:60] if c.heading else None!r}")
        print(f"      {flat[:110]!r} ... {flat[-50:]!r}")
    print()


def main() -> None:
    """Chunk the chosen companies (or all), print the report and show a sample section."""
    tickers = sys.argv[1:] or None
    chunks = list(iter_chunks(tickers))
    report(chunks)
    show(chunks, "MSFT", 2023, "item_1A")


if __name__ == "__main__":
    main()
