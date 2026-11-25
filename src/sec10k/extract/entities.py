"""Tier 3 extraction: find company, product, technology and place names in every chunk with
GLiNER, a local NLP model, and save them per filing.

What it does:
  - `run()` goes through the filings, runs GLiNER (on the Apple GPU, in mini-batches) on the
    text of every chunk, and writes GLiNER's raw output to `data/entities/{accession}.json`.
  - `clean()` turns raw output into usable entities: capitalized names only, no pronouns or
    generic words ("we", "the Company", "our products"), no securities ("common stock"),
    and the filer itself is marked `is_self`.
  - `load_entities()` reads a filing's file back and returns the cleaned entities per chunk.
  - `names_other_company()` answers the route-B question: does a chunk name a company other
    than the filer?

Why it is needed: tier 3 widens the route-B flag from our 130 known companies to any company,
and supplies place (Geography) and product names, all without LLM tokens. Raw output is
stored and cleaning happens on read, so changing a cleaning rule never needs a new GPU run.

Safe to stop and restart: each filing's file is written in one step (temp file + rename).
A filing is skipped when its file exists with the same model settings and the same chunk
hashes, so re-runs only process new or re-chunked filings.

Long runs keep the M3 hot and make it throttle, so run short batches with long pauses:
`--limit N` stops after about N chunks (at the end of a filing). 2,500 chunks is about 10 min
of GPU work; a 10-minute pause after each lets the laptop cool down, e.g.
  caffeinate -i sh -c '.venv/bin/python -m sec10k.extract.entities run --limit 2500 && sleep 600 && ...'

Run: .venv/bin/python -m sec10k.extract.entities run [TICKER ...] [--limit N]
     .venv/bin/python -m sec10k.extract.entities report [TICKER ...]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from itertools import groupby

from sec10k.config import ENTITIES_DIR, load_manifest
from sec10k.extract.companies import CompanyMatcher, load_aliases

MODEL_NAME = "urchade/gliner_medium-v2.1"
LABELS = ["company", "product", "technology", "country or region"]
THRESHOLD = 0.5
BATCH_SIZE = 8
MAX_WORDS = 384  # GLiNER reads at most this many words; longer text is cut off
# Settings stored in each file; a file made with different settings is redone.
SETTINGS = {"model": MODEL_NAME, "labels": LABELS, "threshold": THRESHOLD}

# --------------------------------------------------------------------------------------
# Cleaning rules (applied when reading, so they can change without a new GPU run)
# --------------------------------------------------------------------------------------
_GENERIC = {
    "we", "us", "our", "company", "the company", "companies", "sec", "notes", "covid-19",
    "board", "board of directors", "nasdaq", "nyse", "fasb", "gaap", "irs", "congress",
    "government", "u.s. government", "ai", "ai/ml", "retailers", "customers", "partners",
}
_SECURITIES = re.compile(r"\b(common stock|stock|notes?|debentures|warrants?|shares)\b", re.I)


def _norm(text: str) -> str:
    """Lowercase and keep only letters/digits, so "Digital Ocean, Inc." ~ "DigitalOcean"."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


@dataclass(frozen=True)
class Entity:
    """One cleaned entity found in a chunk."""

    text: str
    label: str       # one of LABELS
    start: int       # character position in the chunk text
    end: int
    score: float     # GLiNER's confidence, 0-1
    is_self: bool    # True if it names the filer itself


class Cleaner:
    """Applies the cleaning rules; knows each company's aliases to spot self-mentions."""

    def __init__(self):
        self.matcher = CompanyMatcher()
        self.self_names: dict[str, list[str]] = {}
        for r in load_aliases().itertuples():
            self.self_names.setdefault(r.ticker, []).append(_norm(r.alias))

    def clean(self, raw: list[dict], filer_ticker: str) -> list[Entity]:
        """Return the entities in `raw` that pass the cleaning rules, with `is_self` set."""
        out = []
        for e in raw:
            text = e["text"].strip()
            low = text.lower()
            if (not re.match(r"[A-Z0-9]", text) or text.isdigit() or low in _GENERIC
                    or low.startswith(("the company", "our "))):
                continue
            if e["label"] == "product" and _SECURITIES.search(text):
                continue
            n = _norm(text)
            is_self = (any(m.is_self for m in self.matcher.find(text, filer_ticker))
                       or any(n.startswith(a) for a in self.self_names.get(filer_ticker, []) if a))
            out.append(Entity(text, e["label"], e["start"], e["end"], e["score"], is_self))
        return out


def names_other_company(entities: list[Entity]) -> bool:
    """True if the chunk names a company other than the filer (the tier-3 part of the route-B flag)."""
    return any(e.label == "company" and not e.is_self for e in entities)


# --------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------
def _path(accession: str):
    """Where a filing's raw entities are stored."""
    return ENTITIES_DIR / f"{accession}.json"


def _is_done(accession: str, hashes: list[str]) -> bool:
    """True if the filing's file exists, used the current settings and matches its current chunks."""
    p = _path(accession)
    if not p.exists():
        return False
    data = json.loads(p.read_text())
    return data.get("settings") == SETTINGS and [c["content_hash"] for c in data["chunks"]] == hashes


def load_entities(accession: str, cleaner: Cleaner | None = None) -> dict[str, list[Entity]]:
    """Return the cleaned entities of one filing, keyed by chunk_id."""
    cleaner = cleaner or Cleaner()
    data = json.loads(_path(accession).read_text())
    return {c["chunk_id"]: cleaner.clean(c["entities"], data["ticker"]) for c in data["chunks"]}


# --------------------------------------------------------------------------------------
# The GPU job
# --------------------------------------------------------------------------------------
def _load_model():
    """Load GLiNER on the Apple GPU (MPS) if available, else the CPU."""
    import torch
    from gliner import GLiNER

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return GLiNER.from_pretrained(MODEL_NAME).to(device), device


def run(tickers: list[str] | None = None, limit: int | None = None) -> None:
    """Run GLiNER on every chunk of filings that are not done yet and save one file per filing.

    Stops after about `limit` chunks (always at the end of a filing), so a long run can be
    split into smaller runs with cool-down pauses in between.
    """
    from sec10k.ingest.chunker import iter_chunks

    ENTITIES_DIR.mkdir(parents=True, exist_ok=True)
    model, device = None, None
    t0, done, filings, skipped, long_chunks = time.time(), 0, 0, 0, 0
    for accession, group in groupby(iter_chunks(tickers), key=lambda c: c.accession):
        chunks = list(group)
        if _is_done(accession, [c.content_hash for c in chunks]):
            skipped += 1
            continue
        if limit is not None and done >= limit:
            break
        if model is None:
            model, device = _load_model()
            print(f"GLiNER on {device}", flush=True)
        texts = [c.text for c in chunks]
        long_chunks += sum(len(t.split()) > MAX_WORDS for t in texts)
        preds = model.inference(texts, LABELS, threshold=THRESHOLD, batch_size=BATCH_SIZE)
        data = {
            "accession": accession, "ticker": chunks[0].ticker, "fiscal_year": chunks[0].fiscal_year,
            "settings": SETTINGS,
            "chunks": [{"chunk_id": c.chunk_id, "content_hash": c.content_hash,
                        "entities": [{"text": e["text"], "label": e["label"], "start": e["start"],
                                      "end": e["end"], "score": round(e["score"], 3)} for e in p]}
                       for c, p in zip(chunks, preds)],
        }
        tmp = _path(accession).with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(_path(accession))  # atomic: the file is either complete or absent
        done += len(chunks)
        filings += 1
        rate = done / (time.time() - t0)
        print(f"  {chunks[0].ticker} FY{chunks[0].fiscal_year}: {len(chunks)} chunks | "
              f"total {done:,} chunks, {filings} filings, {rate:.1f} chunks/s", flush=True)
    print(f"Done: {done:,} chunks in {filings} filings, {(time.time() - t0) / 60:.1f} min; "
          f"{skipped} filings already done; {long_chunks} chunks over {MAX_WORDS} words (cut off)")


def report(tickers: list[str] | None = None) -> None:
    """Print how many filings are done and summary numbers of the cleaned entities."""
    from collections import Counter

    manifest = load_manifest()
    if tickers:
        manifest = manifest[manifest["ticker"].isin(tickers)]
    cleaner = Cleaner()
    n_done = n_chunks = n_flag = 0
    by_label, top = Counter(), {lab: Counter() for lab in LABELS}
    for accession in manifest.accession_number:
        if not _path(accession).exists():
            continue
        n_done += 1
        for ents in load_entities(accession, cleaner).values():
            n_chunks += 1
            n_flag += names_other_company(ents)
            for e in ents:
                if not e.is_self:
                    by_label[e.label] += 1
                    top[e.label][e.text] += 1
    print(f"Filings done: {n_done} / {len(manifest)}  ({n_chunks:,} chunks)")
    if not n_chunks:
        return
    print(f"Chunks naming a company other than the filer: {n_flag:,} ({n_flag / n_chunks:.1%})")
    for lab in LABELS:
        print(f"  {lab}: {by_label[lab]:,} non-self entities; most common: "
              + ", ".join(f"{t} {n}" for t, n in top[lab].most_common(10)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "report"])
    ap.add_argument("tickers", nargs="*", help="only these companies (default: all)")
    ap.add_argument("--limit", type=int, help="run: stop after about this many chunks")
    a = ap.parse_args()
    if a.command == "run":
        run(a.tickers or None, a.limit)
    report(a.tickers or None)
