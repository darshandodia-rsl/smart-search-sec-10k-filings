"""Gives every Chunk node in Neo4j a vector and creates the vector index.

What it does: finds chunks that have no up-to-date vector, embeds them with the local
model, and stores the vector on the chunk. It also creates the Neo4j vector index that
makes "find chunks similar to this question" fast.

Why it is needed: vector search is the base of retrieval. Without these vectors we cannot
find chunks by meaning, and the plain-RAG baseline (which the graph methods are later
compared against) would not exist.

Safe to stop and restart: each round fetches chunks that still need a vector, embeds them,
and saves the vectors in one transaction. A chunk needs a vector if
  - it has none yet, or
  - its text changed since it was embedded (`embedded_hash` differs from `content_hash`), or
  - it was embedded with a different model than the current `EMBED_MODEL`.
If the run stops at any point, running it again continues from there. At most one batch
of work is repeated.

The text that gets embedded is `build_embed_text()`: a short header
("COMPANY (TICKER) 10-K FYyyyy | Item ... | heading") followed by the chunk text. The header
tells the model which company, year and section the chunk comes from.

Run from the project root:  .venv/bin/python -m sec10k.ingest.embed [TICKER ...] [--limit N]
(no tickers = all chunks; --limit stops after about N chunks, which lets a long run be
split into smaller runs with pauses between them)
"""

import argparse
import time

from sec10k.config import EMBED_DIM, EMBED_MODEL
from sec10k.embeddings import get_embed_model
from sec10k.graph.store import get_graph_store
from sec10k.ingest.chunker import build_embed_text

BATCH = 512                    # chunks fetched, embedded and saved per round
INDEX_NAME = "chunk_embedding"  # name of the Neo4j vector index on Chunk.embedding

# Cypher condition (used inside a WHERE) that picks the chunks still needing a vector.
# $t = list of tickers or null for all, $model = the current embedding model name.
NEEDS_VECTOR = """
    ($t IS NULL OR c.ticker IN $t)
    AND (c.embedding IS NULL OR c.embedded_hash IS NULL OR c.embedded_hash <> c.content_hash
         OR c.embedding_model IS NULL OR c.embedding_model <> $model)
"""


def create_vector_index(store) -> None:
    """Create the vector index on Chunk.embedding if it does not exist yet.

    The index uses cosine similarity on 768-dimension vectors, matching our bge model.
    Neo4j fills it in the background as vectors are saved, and search queries use it to
    find similar chunks quickly.
    """
    store.structured_query(f"""
        CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS
        FOR (c:Chunk) ON c.embedding
        OPTIONS {{indexConfig: {{`vector.dimensions`: {EMBED_DIM},
                                `vector.similarity_function`: 'cosine'}}}}
    """)


def run(tickers: list[str] | None = None, limit: int | None = None) -> None:
    """Embed all chunks that need a vector (optionally only some companies).

    With `limit`, stop after about that many chunks (always whole rounds of `BATCH`), so a
    long job can be run in smaller pieces with breaks between them.
    """
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
    """Print how many chunks have a vector, how many are still to do, and the index state.

    Run after `run` to confirm the job finished and the index is fully built (ONLINE, 100%).
    """
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
