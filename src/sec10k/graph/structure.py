"""Writes the structural graph (the skeleton of the data, no LLM) into Neo4j.

The graph shape:

    (Company)-[:FILED]->(Filing)-[:HAS_SECTION]->(Section)-[:HAS_CHUNK]->(Chunk)

So a company has filings, a filing has sections, and a section is cut into chunks.

Why it is needed: this is the backbone of the whole system. Embeddings attach to the Chunk
nodes, extracted entities (competitors, risks, products) will link back to the chunks they
came from, and answers can cite the company, year and section by following these edges.

How each node type is stored:
  Company  A LlamaIndex `EntityNode` (id = company name). It is a real entity, so company
           names mentioned in the text will later be linked to it.
  Chunk    A LlamaIndex `ChunkNode` (text plus metadata as properties). This is the format
           LlamaIndex's PropertyGraphIndex expects, so extraction can attach to it.
  Filing   Written with our own Cypher, and deliberately NOT marked as an entity, so
  Section  entity search never mistakes a filing or section for a company or a risk.

Safe to re-run: every node is written with MERGE by id, so running again updates nodes
instead of duplicating them. If a section now has fewer chunks than before, the extra old
chunks are deleted, so the graph always matches the current chunker output.

Run from the project root:  .venv/bin/python -m sec10k.graph.structure [TICKER ...]
(no tickers = full corpus)
"""

import sys
import time
from itertools import groupby

from llama_index.core.graph_stores.types import ChunkNode, EntityNode

from sec10k.config import load_companies, load_manifest
from sec10k.graph.store import get_graph_store
from sec10k.ingest.chunker import Chunk, iter_chunks
from sec10k.ingest.loader import SECTION_TITLES

BATCH_CHUNKS = 2000     # chunks written per round (about 2 MB of text), to keep memory small


def write_companies(store, companies) -> None:
    """Create or update one Company node per company in `companies`.

    Company nodes are the top of the graph: every filing hangs off one, and company names
    found in the text will later be matched to them.
    """
    store.upsert_nodes([
        EntityNode(name=r.name, label="Company",
                   properties={"cik": r.cik, "ticker": r.ticker, "sic": int(r.sic), "rank": int(r.rank)})
        for r in companies.itertuples()
    ])


def write_filings(store, manifest) -> None:
    """Create or update one Filing node per manifest row, linked from its Company.

    A Filing carries the facts that identify it (accession number, fiscal year, dates), so
    questions like "what changed between 2022 and 2023" can filter by year.
    """
    rows =[{"id": r.accession_number, "company_id": r.name,
             "props": {"accession": r.accession_number, "cik": r.cik, "ticker": r.ticker,
                       "fiscal_year": int(r.fiscal_year), "period_of_report": r.period_of_report,
                       "filing_date": r.filing_date, "form": "10-K"}}
            for r in manifest.itertuples()]
    store.structured_query(
        """
        UNWIND $rows AS row
        MERGE (f:__Node__ {id: row.id})
        SET f:Filing, f += row.props
        WITH f, row
        MATCH (c:__Node__ {id: row.company_id})
        MERGE (c)-[:FILED]->(f)
        """,
        param_map={"rows": rows},
    )


def write_batch(store, chunks: list[Chunk]) -> None:
    """Write a batch of chunks, plus their Section nodes and all the edges between them.

    The batch must contain whole sections (never half of one), because stale chunks are
    cleaned up per section and that needs the section's full chunk list.
    """
    sections = []
    for (acc, sec), group in groupby(chunks, key=lambda c: (c.accession, c.section)):
        group = list(group)
        c0 = group[0]
        sections.append({
            "id": f"{acc}:{sec}", "filing_id": acc, "chunk_ids": [c.chunk_id for c in group],
            "props": {"section": sec, "title": SECTION_TITLES[sec], "ticker": c0.ticker,
                      "fiscal_year": c0.fiscal_year, "n_chunks": len(group)},
        })

    # 1. Section nodes + Filing->Section edges
    store.structured_query(
        """
        UNWIND $rows AS row
        MERGE (s:__Node__ {id: row.id})
        SET s:Section, s += row.props
        WITH s, row
        MATCH (f:__Node__ {id: row.filing_id})
        MERGE (f)-[:HAS_SECTION]->(s)
        """,
        param_map={"rows": sections},
    )
    # 2. Chunk nodes, written through LlamaIndex (vectors are added later by ingest/embed.py)
    store.upsert_nodes([
        ChunkNode(text=c.text, id_=c.chunk_id, properties={
            "cik": c.cik, "ticker": c.ticker, "fiscal_year": c.fiscal_year,
            "filing_date": c.filing_date, "accession": c.accession, "section": c.section,
            "seq": c.seq, "heading": c.heading, "n_tokens": c.n_tokens,
            "content_hash": c.content_hash,
        })
        for c in chunks
    ])
    # 3. Section->Chunk edges, then delete chunks of the section that are no longer produced
    store.structured_query(
        """
        UNWIND $rows AS row
        MATCH (s:__Node__ {id: row.id})
        UNWIND row.chunk_ids AS cid
        MATCH (c:__Node__ {id: cid})
        MERGE (s)-[:HAS_CHUNK]->(c)
        """,
        param_map={"rows": sections},
    )
    store.structured_query(
        """
        UNWIND $rows AS row
        MATCH (:__Node__ {id: row.id})-[:HAS_CHUNK]->(c:Chunk)
        WHERE NOT c.id IN row.chunk_ids
        DETACH DELETE c
        """,
        param_map={"rows": sections},
    )


def build(tickers: list[str] | None = None) -> None:
    """Build the whole structural graph, or only the given companies, then print a report.

    Order: companies, filings, then sections and chunks in batches. This is the entry point
    of the module and the first step of putting data into the graph.
    """
    store = get_graph_store()
    manifest = load_manifest()
    companies = load_companies()
    if tickers:
        manifest = manifest[manifest["ticker"].isin(tickers)]
        companies = companies[companies["ticker"].isin(tickers)]

    t0 = time.time()
    write_companies(store, companies)
    write_filings(store, manifest)
    print(f"{len(companies)} companies, {len(manifest)} filings written")

    batch: list[Chunk] = []
    n_chunks = 0
    for _, filing_chunks in groupby(iter_chunks(tickers), key=lambda c: c.accession):
        batch.extend(filing_chunks)           # add whole filings, so a batch never splits a section
        if len(batch) >= BATCH_CHUNKS:
            write_batch(store, batch)
            n_chunks += len(batch)
            batch = []
            print(f"  {n_chunks:,} chunks  ({time.time() - t0:.0f}s)", flush=True)
    if batch:
        write_batch(store, batch)
        n_chunks += len(batch)
    print(f"Done: {n_chunks:,} chunks in {time.time() - t0:.0f}s\n")
    report(store, tickers)


def report(store, tickers: list[str] | None) -> None:
    """Print node and edge counts, two sanity checks and a few sample paths.

    The checks look for chunks with no section and sections whose chunk count does not
    match what was written. Both should be 0 on a healthy graph. Run after `build` to
    confirm the write worked.
    """
    q = store.structured_query
    p = {"t": tickers}
    flt = "WHERE ($t IS NULL OR n.ticker IN $t)"
    print("Nodes:")
    for label in ["Company", "Filing", "Section", "Chunk"]:
        n = q(f"MATCH (n:{label}) {flt} RETURN count(n) AS n", param_map=p)[0]["n"]
        print(f"  {label:8} {n:,}")
    print("Edges:")
    for rel, a in [("FILED", "Company"), ("HAS_SECTION", "Filing"), ("HAS_CHUNK", "Section")]:
        n = q(f"MATCH (n:{a})-[r:{rel}]->() {flt} RETURN count(r) AS n", param_map=p)[0]["n"]
        print(f"  {rel:12} {n:,}")
    orphans = q(f"MATCH (n:Chunk) {flt} AND NOT ()-[:HAS_CHUNK]->(n) RETURN count(n) AS n", param_map=p)[0]["n"]
    bad = q("""MATCH (n:Section) WHERE $t IS NULL OR n.ticker IN $t
               OPTIONAL MATCH (n)-[:HAS_CHUNK]->(c) WITH n, count(c) AS k
               WHERE k <> n.n_chunks RETURN count(n) AS n""", param_map=p)[0]["n"]
    print(f"Checks: chunks without a section = {orphans}, sections with wrong chunk count = {bad}")
    sample = q("""MATCH (c:Company)-[:FILED]->(f:Filing)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(k:Chunk)
                  WHERE $t IS NULL OR c.ticker IN $t
                  RETURN c.name AS company, f.fiscal_year AS fy, s.title AS section,
                         k.seq AS seq, k.heading AS heading, left(k.text, 80) AS text
                  ORDER BY fy DESC, section, seq LIMIT 3""", param_map=p)
    print("Sample paths:")
    for r in sample:
        print(f"  {r['company']} -> FY{r['fy']} -> {r['section']} -> chunk {r['seq']}: "
              f"[{r['heading']}] {r['text']!r}")


if __name__ == "__main__":
    build(sys.argv[1:] or None)
