"""Writes the extracted facts (the LLM layer of the graph) from the job store into Neo4j.

What it does, for one company at a time:
  1. Reads the company's results from the job store's `results` table: every done item, plus
     every "reuse" item, which takes the result of its source item (identical text in an
     earlier year) but keeps its own fiscal year and chunk. So a paragraph unchanged since
     2016 still gives 2017-2024 their own fact edges, without another LLM call.
  2. Turns them into nodes and edges:
       route B facts  (subject)-[:RELATION]->(object), e.g. (Microsoft)-[:COMPETES_WITH]->(Oracle)
       route A risks  (Company)-[:FACES_RISK]->(Risk {name, category}),
                      plus (Risk)-[:CONCERNS]->(Geography) for places the risk title names
       mentions       (Chunk)-[:MENTIONS]->(entity) for every entity a chunk's facts name
     Fact edges carry fiscal_year, ticker, chunk_ids, evidence, verified and schema_version
     (see FACT_PROPERTIES in schema.py); one edge per (subject, relation, object, fiscal year).
  3. Replaces the company's extracted layer in Neo4j: deletes the company's fact and mention
     edges, writes the new ones, then deletes extracted entity nodes left with no edges.

The filing company itself ("SELF" in the LLM's answer) is linked to its existing Company node
from the structural graph, found by ticker. Every other entity gets a node keyed by its type
and name (`entity_id`); segments are kept per company, because two companies' "Cloud"
segments are different things. Phase 5 (entity resolution) plugs in at `entity_id`.

Why it is needed: the job store's results are the source of truth and the graph is a view
of them. Because each run replaces a company's whole extracted layer, re-running is always
safe, and improving entity resolution later only means re-running this writer. It runs
separately from extraction, so the quota-spending runs never need Neo4j up.

Run: .venv/bin/python -m sec10k.graph.facts [TICKER ...]   (no tickers = every company with results)
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sec10k.config import JOBS_DB, load_companies
from sec10k.extract.llm_extract import _filer_name
from sec10k.graph.schema import ENTITY_TYPES, MENTIONS_RELATION, RELATION_TYPES, SCHEMA_VERSION
from sec10k.graph.store import get_graph_store

_EDGE_TYPES = list(RELATION_TYPES) + [MENTIONS_RELATION]


def tidy_name(name: str) -> str:
    """Tidy a name as the LLM wrote it: every kind of dash becomes "-", spaces are collapsed.

    Models sometimes write typographic dashes (e.g. a non-breaking hyphen in "third‑party");
    without this, the same name would become two different nodes.
    """
    return re.sub(r"\s+", " ", re.sub(r"[‐-―−]", "-", name)).strip()


def entity_id(label: str, name: str, ticker: str, category: str | None = None) -> str:
    """The node id of an extracted entity: its type plus its tidied, lower-cased name.

    Segments also carry the ticker (a segment belongs to one company); risks carry their
    category. This is the one place that decides when two mentions are the same node, so
    Phase 5 entity resolution will replace it with a lookup of canonical names.
    """
    key = tidy_name(name).lower()
    if label == "Segment":
        return f"Segment:{ticker}:{key}"
    if label == "Risk":
        return f"Risk:{category}:{key}"
    return f"{label}:{key}"


class CompanyFacts:
    """The extracted layer of one company, collected in memory before it is written.

    `nodes`: entity id -> (label, display name, extra properties).
    `edges`: (relation, subject id, object id, fiscal year) -> chunk_ids, evidence, verified.
    `mentions`: (chunk id, entity id) pairs.
    """

    def __init__(self, ticker: str, filer_node_id: str, filer_name: str):
        """Start an empty collection for the company with the given structural node id and common name."""
        self.ticker = ticker
        self.filer_node_id = filer_node_id
        self.filer_name = filer_name.lower()
        self.nodes: dict[str, tuple[str, str, dict]] = {}
        self.edges: dict[tuple, dict] = defaultdict(lambda: {"chunk_ids": [], "evidence": [], "verified": False})
        self.mentions: set[tuple[str, str]] = set()

    def node(self, label: str, name: str, chunk_id: str, category: str | None = None) -> str:
        """Register an entity named in a chunk and return its node id (the filer maps to its Company node)."""
        if label == "Company" and name.strip().lower() == self.filer_name:
            return self.filer_node_id
        nid = entity_id(label, name, self.ticker, category)
        if nid not in self.nodes:
            self.nodes[nid] = (label, tidy_name(name), {"category": category} if category else {})
        self.mentions.add((chunk_id, nid))
        return nid

    def edge(self, rel: str, s: str, o: str, fy: int, chunk_id: str, evidence: str, verified: bool) -> None:
        """Add one chunk's support for a fact; facts repeated across chunks of a year share one edge."""
        e = self.edges[(rel, s, o, fy)]
        if chunk_id not in e["chunk_ids"]:
            e["chunk_ids"].append(chunk_id)
            e["evidence"].append(evidence)
        e["verified"] = e["verified"] or verified

    def add_risk(self, payload: dict, fy: int, chunk_id: str) -> None:
        """Add a route-A result: the filer faces the risk, and the risk concerns the places in its title."""
        risk = self.node("Risk", payload["risk"] or payload["title"], chunk_id, payload["category"])
        self.edge("FACES_RISK", self.filer_node_id, risk, fy, chunk_id, payload["title"], True)
        for place in payload.get("places") or []:
            if isinstance(place, str) and place.strip():
                geo = self.node("Geography", place, chunk_id)
                self.edge("CONCERNS", risk, geo, fy, chunk_id, payload["title"], True)

    def add_facts(self, payload: dict, fy: int, chunk_id: str) -> None:
        """Add a route-B result: every validated fact of one chunk."""
        for f in payload["facts"]:
            s = self.node(f["source_type"], f["source"], chunk_id, f.get("category") if f["source_type"] == "Risk" else None)
            o = self.node(f["target_type"], f["target"], chunk_id, f.get("category") if f["target_type"] == "Risk" else None)
            self.edge(f["relation"], s, o, fy, chunk_id, f["evidence"], bool(f.get("verified")))


def collect(db: sqlite3.Connection, ticker: str, filer_node_id: str) -> CompanyFacts:
    """Read the company's results into a CompanyFacts: done items, and reuse items whose source is done.

    A reuse item whose source is waiting to be extracted again (e.g. after a requeue) is left
    out until the source is done, so an old result never reaches the graph.
    """
    cf = CompanyFacts(ticker, filer_node_id, _filer_name(ticker))
    rows = db.execute(
        "SELECT i.route, i.fiscal_year, i.chunk_id, r.payload FROM items i"
        " LEFT JOIN items s ON s.item_id = i.source"
        " JOIN results r ON r.item_id = COALESCE(i.source, i.item_id)"
        " WHERE i.ticker = ? AND ((i.kind = 'extract' AND i.status = 'done')"
        "  OR (i.kind = 'reuse' AND s.status = 'done'))"
        " ORDER BY i.fiscal_year, i.chunk_id", (ticker,))
    for route, fy, chunk_id, payload in rows:
        p = json.loads(payload)
        if route == "A":
            cf.add_risk(p, fy, chunk_id)
        else:
            cf.add_facts(p, fy, chunk_id)
    return cf


def ensure_indexes(store) -> None:
    """Create an index on `ticker` for every fact and mention edge type, so deleting a company's edges is fast."""
    for rel in _EDGE_TYPES:
        store.structured_query(f"CREATE INDEX fact_{rel.lower()}_ticker IF NOT EXISTS FOR ()-[r:{rel}]-() ON (r.ticker)")


def write_company(store, cf: CompanyFacts) -> Counter:
    """Replace the company's extracted layer in Neo4j with `cf`. Returns counts of what was written.

    Labels and relation types come only from the schema, so putting them into the query
    text is safe. Each step is its own query: if one fails, re-running the writer repairs it.
    """
    q, t, out = store.structured_query, cf.ticker, Counter()
    for rel in _EDGE_TYPES:
        q(f"MATCH ()-[r:{rel} {{ticker: $t}}]->() DELETE r", param_map={"t": t})
    by_label: dict[str, list] = defaultdict(list)
    for nid, (label, name, props) in cf.nodes.items():
        by_label[label].append({"id": nid, "name": name, "props": props})
    for label, rows in by_label.items():
        assert label in ENTITY_TYPES, label
        q(f"""UNWIND $rows AS row
              MERGE (e:__Node__ {{id: row.id}})
              ON CREATE SET e.name = row.name
              SET e:__Entity__:{label}, e.origin = 'extraction', e += row.props""", param_map={"rows": rows})
    by_rel: dict[str, list] = defaultdict(list)
    for (rel, s, o, fy), e in cf.edges.items():
        by_rel[rel].append({"s": s, "o": o, "fy": fy, **e})
    for rel, rows in by_rel.items():
        assert rel in RELATION_TYPES, rel
        n = q(f"""UNWIND $rows AS row
                  MATCH (s:__Node__ {{id: row.s}}) MATCH (o:__Node__ {{id: row.o}})
                  MERGE (s)-[r:{rel} {{fiscal_year: row.fy, ticker: $t}}]->(o)
                  SET r.chunk_ids = row.chunk_ids, r.evidence = row.evidence, r.verified = row.verified,
                      r.schema_version = $sv
                  RETURN count(r) AS n""", param_map={"rows": rows, "t": t, "sv": SCHEMA_VERSION})[0]["n"]
        out[rel] += n
        out["edges_missing_node"] += len(rows) - n
    rows = [{"c": c, "e": e} for c, e in cf.mentions]
    n = q(f"""UNWIND $rows AS row
              MATCH (c:__Node__ {{id: row.c}}) MATCH (e:__Node__ {{id: row.e}})
              MERGE (c)-[r:{MENTIONS_RELATION} {{ticker: $t}}]->(e)
              RETURN count(r) AS n""", param_map={"rows": rows, "t": t})[0]["n"] if rows else 0
    out[MENTIONS_RELATION] = n
    out["mentions_missing_chunk"] = len(rows) - n
    out["deleted_orphans"] = q("""MATCH (e:__Entity__ {origin: 'extraction'}) WHERE NOT (e)--()
                                  DELETE e RETURN count(*) AS n""")[0]["n"]
    out["unverified"] = sum(not e["verified"] for e in cf.edges.values())
    return out


def write(tickers: list[str] | None = None, db_path: Path = JOBS_DB) -> None:
    """Write the extracted layer of the given companies (default: all with results) and print a summary."""
    if not db_path.exists():
        sys.exit(f"{db_path} does not exist: no extraction results yet")
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)   # read only: the runner may be writing
    if not tickers:
        tickers = [r[0] for r in db.execute(
            "SELECT DISTINCT i.ticker FROM items i JOIN results r USING (item_id) ORDER BY 1")]
    names = dict(zip(load_companies().ticker, load_companies().name))
    store = get_graph_store()
    ensure_indexes(store)
    for t in tickers:
        out = write_company(store, collect(db, t, names[t]))
        facts = {k: v for k, v in out.items() if k in RELATION_TYPES}
        print(f"{t}: {sum(facts.values())} fact edges {facts}, {out[MENTIONS_RELATION]} mentions, "
              f"{out['unverified']} unverified; missing nodes {out['edges_missing_node']}, "
              f"missing chunks {out['mentions_missing_chunk']}, orphans removed {out['deleted_orphans']}")


if __name__ == "__main__":
    write(sys.argv[1:] or None)
