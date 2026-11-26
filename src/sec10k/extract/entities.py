"""Tier 3 extraction: find company, product, technology and place names in every chunk with
GLiNER, a local NLP model, and save GLiNER's raw output per filing.

What it does: `run()` goes through the filings, runs GLiNER (on the Apple GPU, in mini-batches)
on the text of every chunk, and writes GLiNER's raw output to `data/entities/{accession}.json`.
These raw files are never edited afterwards: they are the reference copy of what the model
found. Cleaning (dropping noise, spotting self-mentions) is done by
`sec10k.extract.entity_cleaning`, which writes separate cleaned files.

Why it is needed: tier 3 widens the route-B flag from our 130 known companies to any company,
and supplies place (Geography) and product names, all without LLM tokens. Keeping the raw
output means cleaning rules can change at any time without another 9+ hour GPU run.

Safe to stop and restart: each filing's file is written in one step (temp file + rename).
A filing is skipped when its file exists with the same model settings and the same chunk
hashes, so re-runs only process new or re-chunked filings.

Long runs keep the M3 hot and make it throttle, so run short batches with long pauses:
`--limit N` stops after about N chunks (at the end of a filing). 2,500 chunks is about 7-8 min
of GPU work; a 7-10 minute pause after each keeps the laptop cool, e.g.
  caffeinate -i sh -c '.venv/bin/python -m sec10k.extract.entities run --limit 2500 && sleep 420 && ...'

Run: .venv/bin/python -m sec10k.extract.entities run [TICKER ...] [--limit N]
     .venv/bin/python -m sec10k.extract.entities status [TICKER ...]
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import groupby

from sec10k.config import ENTITIES_DIR, load_manifest

MODEL_NAME = "urchade/gliner_medium-v2.1"
LABELS = ["company", "product", "technology", "country or region"]
THRESHOLD = 0.5
BATCH_SIZE = 8
MAX_WORDS = 384  # GLiNER reads at most this many words; longer text is cut off
# Settings stored in each file; a file made with different settings is redone.
SETTINGS = {"model": MODEL_NAME, "labels": LABELS, "threshold": THRESHOLD}

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


def status(tickers: list[str] | None = None) -> None:
    """Print how many manifest filings have a raw entity file, i.e. how far the GPU job is."""
    manifest = load_manifest()
    if tickers:
        manifest = manifest[manifest["ticker"].isin(tickers)]
    done = sum(_path(a).exists() for a in manifest.accession_number)
    print(f"Raw entity files: {done} / {len(manifest)} filings")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "status"])
    ap.add_argument("tickers", nargs="*", help="only these companies (default: all)")
    ap.add_argument("--limit", type=int, help="run: stop after about this many chunks")
    a = ap.parse_args()
    if a.command == "run":
        run(a.tickers or None, a.limit)
    status(a.tickers or None)
