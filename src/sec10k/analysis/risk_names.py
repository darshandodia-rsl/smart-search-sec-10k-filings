"""Survey the extracted Risk (and Theme) nodes in Neo4j, before building the risk cluster layer.

What it does (read-only; it never writes to Neo4j):
  1. Reads every Risk node with its category, FACES_RISK edge count, filers and years.
  2. Sharing today: how many risk nodes are already used by more than one company (only
     exact same wording and category share a node now).
  3. Same wording in more than one category ("data breach" as cybersecurity and as data_privacy).
  4. Nearest-neighbour similarity of the risk names (local bge model), within each category,
     with samples per similarity band, so we can see what "the same risk" looks like.
  5. Clustering preview: hierarchical clustering (average linkage) per category at a few
     similarity thresholds: how many clusters, how many span several companies, and samples.
  6. The same quick numbers for Theme nodes.

Why it is needed: risks are the biggest node type (thousands of nodes), and each company words
its risks its own way, so cross-company questions ("which companies cite ransomware?") need
a cluster layer. This survey shows how far apart same-meaning names are and which threshold
groups them well, which decides how the cluster layer is built.

Run: .venv/bin/python -m sec10k.analysis.risk_names
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from sec10k.embeddings import get_embed_model
from sec10k.graph.store import get_graph_store

THRESHOLDS = [0.85, 0.88, 0.90]   # cosine similarity at which names join one cluster (preview)
BANDS = [0.75, 0.80, 0.85, 0.90, 0.95, 1.01]


def load(store, label: str, rel: str) -> pd.DataFrame:
    """Every node of `label` reached by `rel`: id, name, category, edge count, filers, years."""
    rows = store.structured_query(f"""
        MATCH ()-[r:{rel}]->(k:{label})
        RETURN k.id AS id, k.name AS name, coalesce(k.category, '') AS category, count(r) AS edges,
               collect(DISTINCT r.ticker) AS filers, collect(DISTINCT r.fiscal_year) AS years""")
    df = pd.DataFrame(rows)
    df["n_filers"] = df.filers.map(len)
    return df


def embed(names: list[str]) -> np.ndarray:
    """Unit-length bge vectors for the names (cosine similarity = dot product)."""
    v = np.array(get_embed_model().get_text_embedding_batch(names))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def nearest(df: pd.DataFrame, vecs: np.ndarray, within: str | None) -> pd.DataFrame:
    """Each node's most similar other node (within its own category if `within` is the column name)."""
    sim = vecs @ vecs.T
    np.fill_diagonal(sim, -1)
    if within:
        cats = df[within].to_numpy(dtype=object)
        same = cats[:, None] == cats[None, :]
        sim = np.where(same, sim, -1)
    j = sim.argmax(axis=1)
    return pd.DataFrame({"a": df.name.values, "b": df.name.values[j], "sim": sim[np.arange(len(df)), j],
                         "cat": df[within].values if within else ""})


def cluster(df: pd.DataFrame, vecs: np.ndarray, threshold: float) -> np.ndarray:
    """Cluster labels per node: average-linkage clustering inside each category, cut at `threshold` similarity."""
    labels = np.zeros(len(df), dtype=int)
    next_id = 0
    for cat in df.category.unique():
        idx = np.where(df.category.to_numpy(dtype=object) == cat)[0]
        if len(idx) == 1:
            labels[idx] = next_id
            next_id += 1
            continue
        z = linkage(vecs[idx], method="average", metric="cosine")
        local = fcluster(z, t=1 - threshold, criterion="distance")
        labels[idx] = local + next_id
        next_id += local.max() + 1
    return labels


def survey() -> None:
    """Run all survey sections and print the report (see the module docstring)."""
    store = get_graph_store()
    rnd = random.Random(3)
    risks = load(store, "Risk", "FACES_RISK")
    total_filers = len(set().union(*risks.filers))
    print(f"Risk nodes: {len(risks):,} (with FACES_RISK edges), {risks.edges.sum():,} FACES_RISK edges, "
          f"{total_filers} filers, {risks.category.nunique()} categories")

    shared = risks[risks.n_filers > 1]
    print(f"\nShared today (same wording + category): {len(shared):,} nodes used by >1 company "
          f"({len(shared) / len(risks):.0%}); they carry {shared.edges.sum() / risks.edges.sum():.0%} of edges")
    for r in shared.sort_values("n_filers", ascending=False).head(8).itertuples():
        print(f"  {r.n_filers:2} filers  [{r.category}] {r.name}")

    per_cat = risks.groupby("category").agg(nodes=("id", "size"), edges=("edges", "sum")).sort_values("nodes", ascending=False)
    print("\nNodes per category: " + ", ".join(f"{c} {n}" for c, n in per_cat.nodes.items()))

    lower = risks.name.str.lower().str.strip()
    multi = risks.assign(n=lower).groupby("n").category.nunique()
    multi = multi[multi > 1]
    print(f"\nSame wording in >1 category: {len(multi)} names, e.g. "
          + "; ".join(f"{n} ({', '.join(sorted(risks[lower == n].category))})" for n in multi.index[:5]))

    vecs = embed(risks.name.tolist())
    nn = nearest(risks, vecs, "category")
    q = nn.sim.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).to_dict()
    print(f"\nNearest neighbour in the same category, similarity quantiles: {q}")
    for lo, hi in zip(BANDS, BANDS[1:]):
        band = nn[(nn.sim >= lo) & (nn.sim < hi)]
        print(f"  [{lo:.2f}, {min(hi, 1):.2f}): {len(band):,} nodes. Samples:")
        for r in band.iloc[rnd.sample(range(len(band)), min(4, len(band)))].itertuples():
            print(f"      {r.sim:.3f} [{r.cat}] {r.a}  ~  {r.b}")

    print("\nClustering preview (average linkage per category):")
    best = None
    for t in THRESHOLDS:
        labels = cluster(risks, vecs, t)
        c = risks.assign(cl=labels).groupby("cl").agg(
            nodes=("id", "size"), edges=("edges", "sum"), filers=("filers", lambda f: len(set().union(*f))))
        cross = c[c.filers > 1]
        print(f"  sim >= {t}: {len(c):,} clusters (from {len(risks):,} nodes); {len(cross):,} span >1 company "
              f"and hold {cross.edges.sum() / c.edges.sum():.0%} of edges; largest sizes {sorted(c.nodes, reverse=True)[:5]}")
        if t == 0.88:
            best = labels
    print("\nSample cross-company clusters at 0.88 (cybersecurity, competition, talent):")
    r2 = risks.assign(cl=best)
    for cat in ["cybersecurity", "competition", "talent"]:
        cl = r2[r2.category == cat].groupby("cl").filter(lambda g: len(set().union(*g.filers)) > 2)
        for cid in rnd.sample(sorted(cl.cl.unique()), min(2, cl.cl.nunique())):
            g = cl[cl.cl == cid]
            print(f"  [{cat}] {len(g)} names, {len(set().union(*g.filers))} filers: "
                  + " | ".join(g.sort_values("edges", ascending=False).name.head(5)))

    themes = load(store, "Theme", "FOCUSES_ON")
    tv = embed(themes.name.tolist())
    tn = nearest(themes.assign(category=""), tv, None)
    print(f"\nTheme nodes: {len(themes)} (with FOCUSES_ON edges), {(themes.n_filers > 1).sum()} used by >1 company; "
          f"nearest-neighbour similarity median {tn.sim.median():.3f}. Closest pairs:")
    for r in tn.sort_values("sim", ascending=False).drop_duplicates("sim").head(6).itertuples():
        print(f"      {r.sim:.3f} {r.a}  ~  {r.b}")


if __name__ == "__main__":
    survey()
