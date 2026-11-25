"""Plain vector search over all chunks, with year-over-year duplicates grouped.

What it does: embeds a question with the local bge model, asks Neo4j's `chunk_embedding`
vector index for the most similar chunks, and returns them as `Hit` records that carry a
citation (company, fiscal year, section, heading).

Why it is needed: this is the retrieval half of the plain-RAG baseline. The Q&A step sends
these hits to an LLM, and the graph-based retrieval of Phase 6 is measured against it.

Year-over-year grouping: 10-Ks repeat most of their text every year, so a plain top-k
search often returns the same paragraph from five different years and nothing else. Hits
from the same company whose vectors are nearly identical (cosine >= `DUP_THRESHOLD`) are
treated as one passage: the best-scoring one is kept and the years of the others are
listed with it. Hits from different companies are never grouped, because the same wording
at two companies is two separate facts.

Run from the project root:  .venv/bin/python -m sec10k.retrieval.vector_search "question" [-k 10]
"""

import argparse
from dataclasses import dataclass, field


from sec10k.embeddings import get_embed_model
from sec10k.graph.store import get_graph_store
from sec10k.ingest.embed import INDEX_NAME
from sec10k.ingest.loader import SECTION_TITLES

# Two hits from one company with vectors at least this similar count as the same passage.
DUP_THRESHOLD = 0.95
# How many candidates to fetch per result wanted, so enough remain after grouping.
OVERFETCH = 5


@dataclass
class Hit:
    """One retrieved passage plus where it comes from.

    `score` is the cosine similarity to the question (higher = closer). `also_years` lists
    the other fiscal years in which the same company had a near-identical passage; it is
    empty when the passage appeared only once among the candidates.
    """
    chunk_id: str
    score: float
    company: str
    ticker: str
    fiscal_year: int
    section: str
    heading: str | None
    text: str
    also_years: list[int] = field(default_factory=list)

    def citation(self) -> str:
        """Return a short human-readable source label, e.g. "MSFT FY2023, Item 1A Risk Factors"."""
        years = sorted({self.fiscal_year, *self.also_years})
        fy = f"FY{years[0]}" if len(years) == 1 else "FY" + ",".join(str(y) for y in years)
        return f"{self.ticker} {fy}, {SECTION_TITLES[self.section]}"


def search(question: str, k: int = 10) -> list[Hit]:
    """Return up to `k` distinct passages most similar to `question`, best first.

    Fetches `k * OVERFETCH` candidates from the vector index, then groups the
    year-over-year duplicates (see the module doc), so the `k` results are different
    passages rather than copies of one.
    """
    store = get_graph_store()
    vec = get_embed_model().get_query_embedding(question)
    # Neo4j reports cosine rescaled to 0..1 as (1 + cos) / 2; `2 * score - 1` turns it
    # back into plain cosine, so scores and DUP_THRESHOLD use the usual -1..1 scale.
    rows = store.structured_query(f"""
        CALL db.index.vector.queryNodes('{INDEX_NAME}', $n, $vec) YIELD node AS c, score
        MATCH (co:Company)-[:FILED]->(:Filing)-[:HAS_SECTION]->(:Section)-[:HAS_CHUNK]->(c)
        RETURN c.id AS id, 2 * score - 1 AS score, co.name AS company, c.ticker AS ticker,
               c.fiscal_year AS fy, c.section AS section, c.heading AS heading, c.text AS text
        ORDER BY score DESC
    """, param_map={"n": k * OVERFETCH, "vec": vec})
    if not rows:
        return []

    # Similarity of every pair of candidates from the same company, computed inside Neo4j.
    # (The vectors themselves are not fetched: LlamaIndex's query helper drops long lists
    # from results, and comparing them in the database avoids moving them at all.)
    pairs = store.structured_query("""
        UNWIND $ids AS a
        MATCH (x:__Node__ {id: a})
        UNWIND $ids AS b
        MATCH (y:__Node__ {id: b})
        WHERE a < b AND x.ticker = y.ticker
        RETURN a, b, 2 * vector.similarity.cosine(x.embedding, y.embedding) - 1 AS sim
    """, param_map={"ids": [r["id"] for r in rows]})
    similar = {(p["a"], p["b"]) for p in pairs if p["sim"] >= DUP_THRESHOLD}
    similar |= {(b, a) for a, b in similar}

    kept: list[Hit] = []
    for r in rows:  # best score first, so the first of each group is the one kept
        dup = next((h for h in kept if (h.chunk_id, r["id"]) in similar), None)
        if dup:
            if r["fy"] != dup.fiscal_year and r["fy"] not in dup.also_years:
                dup.also_years.append(r["fy"])
            continue
        kept.append(Hit(r["id"], r["score"], r["company"], r["ticker"], r["fy"],
                        r["section"], r["heading"], r["text"]))
    return kept[:k]


def main() -> None:
    """Print the hits for one question from the command line, to inspect retrieval by eye."""
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=10, help="number of distinct passages")
    a = ap.parse_args()
    for i, h in enumerate(search(a.question, a.k), 1):
        print(f"\n[{i}] {h.score:.3f}  {h.citation()}  | {h.heading or '-'}")
        print("    " + h.text[:300].replace("\n", " ") + ("..." if len(h.text) > 300 else ""))


if __name__ == "__main__":
    main()
