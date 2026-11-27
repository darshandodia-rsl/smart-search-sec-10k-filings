"""Group the extracted Risk and Theme nodes into clusters of the same meaning (no LLM, no Neo4j).

What it does, separately for risks and for themes:
  1. Reads every Risk / Theme node from the job store through the graph writer's own `collect`,
     so a node here is a node in the graph (same id). Keeps its name, category (risks), how
     many edges use it and which companies use it.
  2. Embeds the names with the local bge model and clusters them with average-linkage
     hierarchical clustering: two groups join while the average cosine similarity between
     their names is at least CLUSTER_SIMILARITY (0.85). Risks are clustered across all
     categories, because the LLM sometimes gives the same risk two different categories.
  3. Names each cluster after its most central member (the name closest to the cluster's
     average meaning) and gives it the category most of its edges use.
  Writes `data/resolution/clusters.csv` (one row per node: its cluster id, name, category,
  size, companies) and prints a report.

Why it is needed: each company words its risks its own way ("security breach exposing
customer data", "Security breach of customer and employee data"), so a risk node is almost
never shared and cross-company questions find nothing. The graph writer (`graph.facts`) adds
a RiskCluster / ThemeCluster node per cluster with IN_CLUSTER edges from its members; the
member nodes stay as written. Clusters are rebuilt from scratch on every run, so cluster ids
can change when new filings arrive; the graph is a view and is rewritten from this file.

Run: .venv/bin/python -m sec10k.resolve.clusters
"""

from __future__ import annotations

import sqlite3
from collections import Counter

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from sec10k.config import CLUSTERS, JOBS_DB, RESOLUTION_DIR, load_companies
from sec10k.graph.facts import collect

CLUSTER_SIMILARITY = 0.85            # average cosine similarity at which names share a cluster
CLUSTERED_TYPES = ("Risk", "Theme")  # node type -> cluster node type "<type>Cluster"


def node_inventory(tickers: list[str] | None = None) -> pd.DataFrame:
    """Every Risk/Theme node in the job store: id, type, name, category, edge count, companies using it."""
    db = sqlite3.connect(f"file:{JOBS_DB}?mode=ro", uri=True)   # read only: a runner may be writing
    if not tickers:
        tickers = [r[0] for r in db.execute(
            "SELECT DISTINCT i.ticker FROM items i JOIN results r USING (item_id) ORDER BY 1")]
    filer_nodes = dict(zip(load_companies().ticker, load_companies().name))
    rows: dict[str, dict] = {}
    for t in tickers:
        cf = collect(db, t, filer_nodes[t])
        ends = Counter()
        for rel, s, o, fy in cf.edges:
            ends[s] += 1
            ends[o] += 1
        for nid, (label, name, props) in cf.nodes.items():
            if label not in CLUSTERED_TYPES:
                continue
            r = rows.setdefault(nid, {"node_id": nid, "type": label, "name": name,
                                      "category": props.get("category", ""), "edges": 0, "filers": set()})
            r["edges"] += ends[nid]
            r["filers"].add(t)
    return pd.DataFrame(rows.values())


def embed(names: list[str]) -> np.ndarray:
    """Unit-length bge vectors for the names (identical names are embedded once)."""
    from sec10k.embeddings import get_embed_model

    unique = sorted(set(names))
    vecs = np.array(get_embed_model().get_text_embedding_batch(unique))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    pos = {n: i for i, n in enumerate(unique)}
    return vecs[[pos[n] for n in names]]


def cluster_nodes(df: pd.DataFrame, threshold: float = CLUSTER_SIMILARITY) -> pd.DataFrame:
    """Add cluster columns to the nodes of one type (see the module docstring for the method)."""
    vecs = embed(df.name.tolist())
    if len(df) > 1:
        labels = fcluster(linkage(vecs, method="average", metric="cosine"), t=1 - threshold, criterion="distance")
    else:
        labels = np.ones(len(df), dtype=int)
    df = df.assign(cl=labels)
    kind = df.type.iloc[0]
    out = []
    for _, g in df.groupby("cl"):
        idx = g.index.to_numpy()
        centre = vecs[idx].mean(axis=0)
        central = g.loc[idx[np.argmax(vecs[idx] @ centre)]]
        cats = Counter()
        for r in g.itertuples():
            cats[r.category] += r.edges or 1
        filers = set().union(*g.filers)
        out.append(g.assign(cluster_id=f"{kind}Cluster:{central['name'].lower()}", cluster_name=central["name"],
                            cluster_category=cats.most_common(1)[0][0], cluster_size=len(g),
                            cluster_filers=len(filers)))
    return pd.concat(out).drop(columns="cl")


def build() -> pd.DataFrame:
    """Cluster risks and themes, write the clusters file, print a report and return the table."""
    nodes = node_inventory()
    parts = [cluster_nodes(nodes[nodes.type == t].reset_index(drop=True)) for t in CLUSTERED_TYPES
             if (nodes.type == t).any()]
    out = pd.concat(parts, ignore_index=True)
    RESOLUTION_DIR.mkdir(parents=True, exist_ok=True)
    out.assign(filers=out.filers.map(lambda f: " ".join(sorted(f))))[
        ["type", "node_id", "name", "category", "edges", "filers", "cluster_id", "cluster_name",
         "cluster_category", "cluster_size", "cluster_filers"]].to_csv(CLUSTERS, index=False)

    for t in CLUSTERED_TYPES:
        d = out[out.type == t]
        c = d.drop_duplicates("cluster_id")
        cross = d[d.cluster_filers > 1]
        print(f"{t}: {len(d):,} nodes -> {len(c):,} clusters (similarity >= {CLUSTER_SIMILARITY}); "
              f"{(c.cluster_filers > 1).sum():,} clusters span >1 company and hold {cross.edges.sum() / d.edges.sum():.0%} "
              f"of edges (exact-wording nodes shared today: {(d.filers.map(len) > 1).sum():,})")
        print("  Most widely shared: " + "; ".join(
            f"{r.cluster_name} ({r.cluster_filers} cos, {r.cluster_size} wordings)"
            for r in c.sort_values(["cluster_filers", "cluster_size"], ascending=False).head(6).itertuples()))
    print(f"\nWritten: {CLUSTERS}")
    return out


if __name__ == "__main__":
    build()
