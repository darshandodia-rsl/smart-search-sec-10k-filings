"""Phase 2.4: embed every Chunk node in Neo4j and build the vector index.

Resumable and never pays twice: each round fetches chunks that still need a vector,
embeds them, and writes the vectors back in one transaction. A chunk needs a vector if
  - it has none, or
  - its text changed since it was embedded (embedded_hash <> content_hash), or
  - it was embedded with a different model (embedding_model <> EMBED_MODEL).
Stop or crash at any time: re-running continues where it left off; at most one batch
is redone.

The text embedded is build_embed_text(): "COMPANY (TICKER) 10-K FYyyyy | Item ... | heading"
+ chunk text, rebuilt from the chunk's properties so it matches what the chunker defined.

Run from the project root:  .venv/bin/python -m sec10k.ingest.embed [TICKER ...] [--limit N]
(no tickers = all chunks; --limit = stop after N chunks, e.g. to let the machine cool down)
"""

import argparse
import time

from sec10k.config import EMBED_DIM, EMBED_MODEL
from sec10k.embeddings import get_embed_model
from sec10k.graph.store import get_graph_store
from sec10k.ingest.chunker import build_embed_text

BATCH = 512                    # chunks per fetch/write round
INDEX_NAME = "chunk_embedding"

NEEDS_VECTOR = """
    ($t IS NULL OR c.ticker IN $t)
    AND (c.embedding IS NULL OR c.embedded_hash IS NULL OR c.embedded_hash <> c.content_hash
         OR c.embedding_model IS NULL OR c.embedding_model <> $model)
"""


def create_vector_index(store) -> None:
    """Cosine similarity on normalized 768-dim bge vectors. Neo4j fills the index in the
    background as vectors are written."""
    store.structured_query(f"""
        CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS
        FOR (c:Chunk) ON c.embedding
        OPTIONS {{indexConfig: {{`vector.dimensions`: {EMBED_DIM},
                                `vector.similarity_function`: 'cosine'}}}}
    """)


def run(tickers: list[str] | None = None, limit: int | None = None) -> None:
    """Embed chunks that need a vector; with `limit`, stop after about that many
    (whole rounds of BATCH), e.g. to run the corpus in cool-down-separated batches."""
    store = get_graph_store()
    create_vector_index(store)
    params = {"t": tickers, "model": EMBED_MODEL}
    todo = store.structured_query(
        f"MATCH (c:Chunk) WHERE {NEEDS_VECTOR} RETURN count(c) AS n", param_map=params)[0]["n"]
    print(f"Chunks needing a vector: {todo:,}")
    if not todo:
        return
    if limit:
        todo = min(todo, limit)
        print(f"This run: up to {todo:,}")
    model = get_embed_model()

    done, t0 = 0, time.time()
    while True:
        if done >= todo:
            break
        rows = store.structured_query(f"""
            MATCH (co:Company)-[:FILED]->(:Filing)-[:HAS_SECTION]->(:Section)-[:HAS_CHUNK]->(c:Chunk)
            WHERE {NEEDS_VECTOR}
            RETURN c.id AS id, c.content_hash AS hash, co.name AS company, c.ticker AS ticker,
                   c.fiscal_year AS fy, c.section AS section, c.heading AS heading, c.text AS text
            LIMIT $n
        """, param_map={**params, "n": min(BATCH, todo - done)})
        if not rows:
            break
        texts = [build_embed_text(r["company"], r["ticker"], r["fy"], r["section"],
                                  r["heading"], r["text"]) for r in rows]
        vectors = model.get_text_embedding_batch(texts)
        store.structured_query("""
            UNWIND $rows AS row
            MATCH (c:__Node__ {id: row.id})
            CALL db.create.setNodeVectorProperty(c, 'embedding', row.vector)
            SET c.embedded_hash = row.hash, c.embedding_model = $model
        """, param_map={"rows": [{"id": r["id"], "hash": r["hash"], "vector": v}
                                 for r, v in zip(rows, vectors)], "model": EMBED_MODEL})
        done += len(rows)
        rate = done / (time.time() - t0)
        print(f"  {done:,}/{todo:,}  {rate:.0f} chunks/s  ETA {(todo - done) / rate / 60:.0f} min", flush=True)
    print(f"Done: {done:,} chunks in {(time.time() - t0) / 60:.1f} min")


def report(tickers: list[str] | None = None) -> None:
    store = get_graph_store()
    p = {"t": tickers, "model": EMBED_MODEL}
    r = store.structured_query(f"""
        MATCH (c:Chunk) WHERE $t IS NULL OR c.ticker IN $t
        RETURN count(c) AS chunks, count(c.embedding) AS with_vector,
               sum(CASE WHEN {NEEDS_VECTOR} THEN 1 ELSE 0 END) AS still_todo
    """, param_map=p)[0]
    idx = store.structured_query(
        "SHOW VECTOR INDEXES YIELD name, state, populationPercent WHERE name = $n RETURN *",
        param_map={"n": INDEX_NAME})
    print(f"Chunks: {r['chunks']:,}  with vector: {r['with_vector']:,}  still to do: {r['still_todo']:,}")
    print(f"Index: {idx}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*", help="only these companies (default: all)")
    ap.add_argument("--limit", type=int, help="embed at most this many chunks, then stop")
    a = ap.parse_args()
    run(a.tickers or None, a.limit)
    report(a.tickers or None)
