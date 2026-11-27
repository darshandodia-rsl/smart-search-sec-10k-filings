"""Stage 4 of entity resolution: find the unclear pairs of names that need a judgement.

What it does, after stages 1-3 (`names`):
  - Prefix pairs: one key is the start of another, word by word ("cisco" / "cisco systems",
    "microsoft" / "microsoft azure", "amazon" / "amazon web services"). These catch short vs
    full names, products named after their maker, and subsidiaries.
  - Embedding pairs: keys whose names mean nearly the same to the local bge model (cosine
    similarity at or above a floor), among each key's 5 nearest neighbours. These catch
    spellings rules miss ("Amazon Web Service" / "Amazon Web Services").
  - Hand overrides are added as pairs that already carry their decision.
  - Pairs between two corpus companies are left out: the alias table already knows them.
  - Marks the pairs the LLM judge will see (`to_llm`): every pair with a Company side (prefix,
    or similarity >= 0.85) and Product-Product pairs only when near-identical (>= 0.95).
  Writes all pairs to `data/resolution/candidate_pairs.csv` (with similarity, source and
  `to_llm`) and prints a report: counts per source and similarity band, and samples.

Why it is needed: the LLM judge (next stage) should only see pairs that are really unclear.
This step is free and local, and its counts tell how much LLM work is left before any is
spent. Company pairs carry the cross-company links; most Product-Product pairs are one
filer's own distinct products, so only near-identical spellings of those are worth judging.

Run: .venv/bin/python -m sec10k.resolve.candidates [TICKER ...]   (default: every company with results)
"""

from __future__ import annotations

import random
import sys

import numpy as np
import pandas as pd

from sec10k.config import RESOLUTION_DIR
from sec10k.resolve.names import Key, load_overrides, match_aliases, name_inventory

EMBED_FLOOR = 0.80   # lowest similarity kept in the pairs file
# Which pairs the LLM judges (`to_llm`): a pair with a Company side if it is a prefix pair or has
# similarity >= COMPANY_FLOOR; a Product-Product pair only if near-identical (>= PRODUCT_FLOOR).
COMPANY_FLOOR = 0.85
PRODUCT_FLOOR = 0.95
NEIGHBOURS = 5       # nearest neighbours per key that are considered
BANDS = [0.80, 0.85, 0.90, 0.95, 1.01]
# Single words too common to make a prefix pair on their own ("cloud" / "cloud software group").
GENERIC_PREFIXES = {"the", "cloud", "global", "national", "american", "united", "international",
                    "first", "general", "digital", "data", "open", "new", "other", "us", "u", "a"}


def prefix_pairs(keys: dict[str, Key]) -> list[tuple[str, str]]:
    """Pairs (longer key, shorter key) where the shorter one is the first word(s) of the longer one."""
    out = []
    for k in keys:
        words = k.split()
        for i in range(1, len(words)):
            p = " ".join(words[:i])
            if p in keys and not (i == 1 and (p in GENERIC_PREFIXES or len(p) < 2)):
                out.append((k, p))
    return out


def similarities(keys: dict[str, Key]) -> tuple[list[str], np.ndarray]:
    """Embed every key's display name with the local model; return the key order and the cosine-similarity matrix."""
    from sec10k.embeddings import get_embed_model

    order = list(keys)
    vecs = np.array(get_embed_model().get_text_embedding_batch([keys[k].name for k in order]))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return order, vecs @ vecs.T


def embedding_pairs(order: list[str], sim: np.ndarray, keys: dict[str, Key]) -> list[tuple[str, str, float]]:
    """Pairs among each key's nearest neighbours with similarity >= EMBED_FLOOR; the less-used key comes first."""
    out = {}
    for i, k in enumerate(order):
        row = sim[i].copy()
        row[i] = -1
        for j in np.argsort(-row)[:NEIGHBOURS]:
            if row[j] < EMBED_FLOOR:
                break
            a, b = sorted([k, order[j]], key=lambda x: (keys[x].edges, x))
            out[(a, b)] = float(row[j])
    return [(a, b, s) for (a, b), s in out.items()]


def build_pairs(keys: dict[str, Key]) -> pd.DataFrame:
    """All candidate pairs from stages 2 and 4, one row per unordered pair, without corpus-vs-corpus pairs."""
    order, sim = similarities(keys)
    pos = {k: i for i, k in enumerate(order)}
    rows, seen = [], set()

    def add(a, b, source, decision="", note=""):
        if frozenset((a, b)) in seen or a == b:
            return
        seen.add(frozenset((a, b)))
        ka, kb = keys.get(a), keys.get(b)
        if ka and kb and ka.ticker and kb.ticker and source != "override":
            return
        rows.append({"a": a, "b": b, "source": source, "decision": decision, "note": note,
                     "similarity": round(float(sim[pos[a], pos[b]]), 3) if ka and kb else None,
                     "a_name": ka.name if ka else a, "b_name": kb.name if kb else b,
                     "a_label": ka.label if ka else "", "b_label": kb.label if kb else "",
                     "a_ticker": (ka.ticker if ka else "") or "", "b_ticker": (kb.ticker if kb else "") or "",
                     "a_edges": ka.edges if ka else 0, "b_edges": kb.edges if kb else 0})

    for (a, b), (decision, note) in load_overrides().items():
        add(a, b, "override", decision, note)
    for a, b in prefix_pairs(keys):
        add(a, b, "prefix")
    for a, b, _ in embedding_pairs(order, sim, keys):
        add(a, b, "embedding")
    df = pd.DataFrame(rows)
    df["to_llm"] = llm_scope(df)
    return df


def llm_scope(df: pd.DataFrame) -> pd.Series:
    """True for the pairs the LLM judge should see (the rule written next to COMPANY_FLOOR); pairs with a hand decision never go."""
    has_company = (df.a_label == "Company") | (df.b_label == "Company")
    sim = df.similarity.astype(float)
    company = has_company & ((df.source == "prefix") | (sim >= COMPANY_FLOOR))
    product = ~has_company & (sim >= PRODUCT_FLOOR)
    return (company | product) & (df.decision == "")


def report(tickers: list[str] | None = None) -> None:
    """Run stages 1-4, write the pairs file and print the counts and samples (see the module docstring)."""
    keys = name_inventory(tickers=tickers)
    n_spellings = sum(len(k.variants) for k in keys.values())
    clashes = [k for k in keys.values() if k.by_label["Company"] and k.by_label["Product"]]
    print(f"Stage 1 (rules): {n_spellings:,} spellings -> {len(keys):,} keys "
          f"({sum(k.label == 'Company' for k in keys.values()):,} Company, "
          f"{sum(k.label == 'Product' for k in keys.values()):,} Product by majority); "
          f"{len(clashes)} keys used as both types")
    n_alias = match_aliases(keys)
    print(f"Stage 3 (alias table): {n_alias} keys -> {len({k.ticker for k in keys.values() if k.ticker})} "
          f"corpus companies")
    print(f"Stage 2 (overrides): {len(load_overrides())} hand decisions")

    pairs = build_pairs(keys)
    RESOLUTION_DIR.mkdir(parents=True, exist_ok=True)
    out = RESOLUTION_DIR / "candidate_pairs.csv"
    pairs.to_csv(out, index=False)

    print(f"\nStage 4: {len(pairs):,} candidate pairs (corpus-vs-corpus pairs left out)")
    for src, g in pairs.groupby("source"):
        combos = (g.a_label.str[0] + "-" + g.b_label.str[0]).value_counts().to_dict()
        print(f"  {src}: {len(g):,}  by type {combos}")
    emb = pairs[pairs.source == "embedding"]
    print("\nEmbedding pairs by similarity band (prefix pairs not repeated here):")
    rnd = random.Random(7)
    for lo, hi in zip(BANDS, BANDS[1:]):
        band = emb[(emb.similarity >= lo) & (emb.similarity < hi)]
        print(f"  [{lo:.2f}, {min(hi, 1):.2f}{']' if hi > 1 else ')'}: {len(band):,} pairs. Samples:")
        for r in band.iloc[rnd.sample(range(len(band)), min(6, len(band)))].itertuples():
            print(f"      {r.similarity:.3f}  {r.a_name} ({r.a_label[0]}) ~ {r.b_name} ({r.b_label[0]})")
    print("\nPrefix pair samples:")
    pre = pairs[pairs.source == "prefix"]
    for r in pre.iloc[rnd.sample(range(len(pre)), min(12, len(pre)))].itertuples():
        print(f"      {r.a_name} ({r.a_label[0]}) -> {r.b_name} ({r.b_label[0]}{', ' + r.b_ticker if r.b_ticker else ''})")
    llm = pairs[pairs.to_llm]
    has_c = (llm.a_label == "Company") | (llm.b_label == "Company")
    print(f"\nPairs the LLM will judge (to_llm): {len(llm):,} ≈ {-(-len(llm) // 25)} calls of 25: "
          f"{int(has_c.sum()):,} with a Company side (prefix or similarity >= {COMPANY_FLOOR}), "
          f"{int((~has_c).sum()):,} Product-Product (similarity >= {PRODUCT_FLOOR})")
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    report(sys.argv[1:] or None)
