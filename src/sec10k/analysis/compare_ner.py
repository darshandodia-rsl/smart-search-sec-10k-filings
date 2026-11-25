"""Compare two local NLP models (spaCy vs GLiNER) for extraction tier 3 on a fixed chunk sample.

What it does: takes the same ~300 chunks (random, fixed seed, from a dozen varied companies),
runs both models on them in mini-batches, and reports for each:
  - speed, projected to the full corpus (measured on a cool machine, so a lower bound),
  - how many entities of each kind it finds (company, product, technology, place),
  - how many chunks it would flag for LLM route B (names a company other than the filer),
    compared with the tier-2 flag (corpus-company names + relationship cue phrases),
  - random entities with their context, for a hand check of precision.
All entities are also saved to `data/profile/ner_compare.csv` for review.

Why it is needed: tier 3 should widen the route-B flag to any company (not just our 130) and
find places (Geography) and maybe products (route C). This study picks which model to use
before running one over all 170K chunks.

Run: .venv/bin/python -m sec10k.analysis.compare_ner
Needs spaCy and its model, which the project does not install (GLiNER is the tier-3 model):
  .venv/bin/pip install spacy==3.8.18 && .venv/bin/python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import random
import time
from collections import Counter

import pandas as pd

from sec10k.config import PROFILE_DIR
from sec10k.extract.companies import CompanyMatcher
from sec10k.extract.patterns import find_cues
from sec10k.ingest.chunker import iter_chunks

SAMPLE_TICKERS = ["MSFT", "AAPL", "CRM", "TWLO", "MBLY", "PANW", "TTWO", "LDOS", "MTCH",
                  "OKTA", "DOCN", "TOST"]
SAMPLE_SIZE = 300
CORPUS_CHUNKS = 170_187
GLINER_MODEL = "urchade/gliner_medium-v2.1"
GLINER_LABELS = ["company", "product", "technology", "country or region"]
# spaCy's built-in entity types, mapped to the same four kinds (spaCy has no "technology").
SPACY_LABELS = {"ORG": "company", "PRODUCT": "product", "GPE": "country or region",
                "LOC": "country or region"}


def sample_chunks():
    """Return the fixed random sample of chunks used for both models."""
    chunks = list(iter_chunks(SAMPLE_TICKERS))
    return random.Random(0).sample(chunks, SAMPLE_SIZE)


def run_spacy(texts: list[str]) -> tuple[list[list[dict]], float]:
    """Run spaCy's small English model (NER only) and return entities per text and seconds taken."""
    import spacy

    nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "attribute_ruler", "lemmatizer"])
    t = time.time()
    out = []
    for doc in nlp.pipe(texts, batch_size=64):
        out.append([{"text": e.text, "label": SPACY_LABELS[e.label_], "start": e.start_char,
                     "end": e.end_char, "score": None}
                    for e in doc.ents if e.label_ in SPACY_LABELS])
    return out, time.time() - t


def run_gliner(texts: list[str]) -> tuple[list[list[dict]], float, str]:
    """Run GLiNER with our own labels on the Apple GPU if possible; return entities, seconds, device."""
    import torch
    from gliner import GLiNER

    model = GLiNER.from_pretrained(GLINER_MODEL)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.inference(texts[:8], GLINER_LABELS, batch_size=8)  # warm-up, not timed
    t = time.time()
    preds = model.inference(texts, GLINER_LABELS, threshold=0.5, batch_size=8)
    secs = time.time() - t
    out = [[{"text": e["text"], "label": e["label"], "start": e["start"], "end": e["end"],
             "score": round(e["score"], 3)} for e in p] for p in preds]
    return out, secs, device


def main() -> None:
    """Run both models on the sample and print the comparison."""
    chunks = sample_chunks()
    texts = [c.text for c in chunks]
    matcher = CompanyMatcher()
    long = sum(len(t.split()) > 384 for t in texts)
    print(f"Sample: {len(chunks)} chunks from {len(SAMPLE_TICKERS)} companies; "
          f"sections {dict(Counter(c.section for c in chunks))}; {long} over GLiNER's 384-word window\n")

    tier2 = [any(q.named for q in find_cues(c.text)) or
             any(not m.is_self for m in matcher.find(c.text, c.ticker)) for c in chunks]

    sp, sp_secs = run_spacy(texts)
    gl, gl_secs, device = run_gliner(texts)

    rows = []
    for tool, preds in [("spacy", sp), ("gliner", gl)]:
        for c, ents in zip(chunks, preds):
            for e in ents:
                # An entity is "self" if the alias table says it names the filer itself.
                is_self = any(m.is_self for m in matcher.find(e["text"], c.ticker))
                ctx = c.text[max(0, e["start"] - 50):e["end"] + 50].replace("\n", " ")
                rows.append({"tool": tool, "chunk_id": c.chunk_id, "ticker": c.ticker,
                             "section": c.section, "label": e["label"], "text": e["text"],
                             "score": e["score"], "is_self": is_self, "context": ctx})
    df = pd.DataFrame(rows)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROFILE_DIR / "ner_compare.csv", index=False)

    print(f"{'':28}{'spaCy':>14}{'GLiNER':>14}")
    print(f"{'seconds for sample':28}{sp_secs:>14.1f}{gl_secs:>14.1f}   (GLiNER on {device})")
    print(f"{'chunks / second':28}{len(texts) / sp_secs:>14.1f}{len(texts) / gl_secs:>14.1f}")
    print(f"{'projected full corpus':28}{CORPUS_CHUNKS / (len(texts) / sp_secs) / 60:>11.0f} min"
          f"{CORPUS_CHUNKS / (len(texts) / gl_secs) / 60:>11.0f} min")
    for label in GLINER_LABELS:
        n = [len(df[(df.tool == t) & (df.label == label)]) for t in ("spacy", "gliner")]
        print(f"{'entities: ' + label:28}{n[0]:>14}{n[1]:>14}")

    print(f"\nRoute-B flag (chunk names a company other than the filer), of {len(chunks)} chunks:")
    print(f"  tier 2 (names + cues):  {sum(tier2)}")
    for tool in ("spacy", "gliner"):
        d = df[(df.tool == tool) & (df.label == "company") & ~df.is_self]
        flagged = {cid for cid in d.chunk_id}
        both = sum(1 for c, f in zip(chunks, tier2) if f and c.chunk_id in flagged)
        print(f"  {tool:7} company entity:  {len(flagged)}  (also tier 2: {both}; "
              f"tier 2 OR {tool}: {sum(tier2) + len(flagged) - both})")

    rng = random.Random(1)
    for tool in ("spacy", "gliner"):
        for label in GLINER_LABELS:
            d = df[(df.tool == tool) & (df.label == label) & ~df.is_self]
            if d.empty:
                continue
            top = Counter(d.text).most_common(12)
            print(f"\n===== {tool} · {label}: {len(d)} (non-self). Most common: "
                  + ", ".join(f"{t} {n}" for t, n in top))
            for r in d.iloc[rng.sample(range(len(d)), min(8, len(d)))].itertuples():
                print(f"   [{r.ticker}] «{r.text}»  …{r.context}…")


if __name__ == "__main__":
    main()
