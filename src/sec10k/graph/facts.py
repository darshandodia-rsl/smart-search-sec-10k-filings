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
from the structural graph, found by ticker. Company and Product names are looked up in the
entity map (`data/resolution/entity_map.csv`, built by `resolve.entity_map`) by their rule
key: every spelling of one real thing becomes one node with the canonical name and type, and
a corpus company becomes its structural Company node. Names not in the map (e.g. from a
company extracted after the map was built) and other types get a node keyed by their type and
name (`entity_id`). Places are first given their canonical name (`geography_name`: the
hand table `resolve/geography_aliases.csv`, and "N countries" -> "Multiple countries"), so
"U.S." and "United States" are one Geography node. Segments are kept per company, because two companies' "Cloud" segments
are different things. After all companies, the entity links (`entity_links.csv`) are written
as PRODUCT_OF / SUBSIDIARY_OF edges, and the risk/theme clusters (`clusters.csv`, built by
`resolve.clusters`) as RiskCluster / ThemeCluster nodes with IN_CLUSTER edges from their
members (all origin "resolution"). A mapped node's type may differ
from the type the LLM gave the name, so a few edges join types the extraction schema does not
list (e.g. a name extracted as a Product that is really a Company).

Why it is needed: the job store's results are the source of truth and the graph is a view
of them. Because each run replaces a company's whole extracted layer, re-running is always
safe, and improving entity resolution later only means rebuilding the map and re-running
this writer. It runs
separately from extraction, so the quota-spending runs never need Neo4j up.

Run: .venv/bin/python -m sec10k.graph.facts [TICKER ...]   (no tickers = every company with results)
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

from sec10k.config import CLUSTERS, ENTITY_LINKS, ENTITY_MAP, GEOGRAPHY_ALIASES, JOBS_DB, load_companies
from sec10k.extract.llm_extract import filer_name
from sec10k.graph.schema import ENTITY_TYPES, MENTIONS_RELATION, RELATION_TYPES, SCHEMA_VERSION_BY_PROMPT
from sec10k.graph.store import get_graph_store
from sec10k.resolve.keys import RESOLVED_TYPES, corpus_product_id, rule_key

_EDGE_TYPES = list(RELATION_TYPES) + [MENTIONS_RELATION]
# Edges written from entity resolution, not from extraction (they are not in the extraction schema).
LINK_RELATIONS = {
    "PRODUCT_OF": "(Product)-[:PRODUCT_OF]->(Company): the product is made by the company",
    "SUBSIDIARY_OF": "(Company)-[:SUBSIDIARY_OF]->(Company): the company is owned by the other",
}
CLUSTER_RELATION = "IN_CLUSTER"   # (Risk)-[:IN_CLUSTER]->(RiskCluster), (Theme)-[:IN_CLUSTER]->(ThemeCluster)
CLUSTER_LABELS = ("RiskCluster", "ThemeCluster")
# Edges that only entity resolution writes; they alone never keep an extracted node alive.
RESOLUTION_EDGES = list(LINK_RELATIONS) + [CLUSTER_RELATION]


def tidy_name(name: str) -> str:
    """Tidy a name as the LLM wrote it: every kind of dash becomes "-", spaces are collapsed.

    Models sometimes write typographic dashes (e.g. a non-breaking hyphen in "third‑party");
    without this, the same name would become two different nodes.
    """
    return re.sub(r"\s+", " ", re.sub(r"[‐-―−]", "-", name)).strip()


def entity_id(label: str, name: str, ticker: str, category: str | None = None) -> str:
    """The node id of an extracted entity: its type plus its tidied, lower-cased name.

    Segments also carry the ticker (a segment belongs to one company); risks carry their
    category. Used for every name the entity map does not cover (see `CompanyFacts.node`).
    """
    key = tidy_name(name).lower()
    if label == "Segment":
        return f"Segment:{ticker}:{key}"
    if label == "Risk":
        return f"Risk:{category}:{key}"
    return f"{label}:{key}"


MULTIPLE_COUNTRIES = "Multiple countries"
# "80 countries", "over 70 countries", "approximately 200 countries and territories"
_COUNT_OF_COUNTRIES = re.compile(r"^(?:approximately|about|over|more than|nearly)?\s*[\d,]+\+?\s+countries"
                                 r"(?:\s+and\s+territories)?$", re.I)


@cache
def _geography_aliases() -> dict[str, str]:
    """The hand table of place spellings as {rule key: canonical place}; read once per process."""
    if not GEOGRAPHY_ALIASES.exists():
        return {}
    with GEOGRAPHY_ALIASES.open() as f:
        return {rule_key(r["name"]): r["canonical"] for r in csv.DictReader(f)}


def geography_name(name: str) -> str:
    """The canonical name of a place: "U.S." -> "United States", "80 countries" -> "Multiple countries".

    Spellings not in the hand table and not a count of countries are returned unchanged.
    """
    if _COUNT_OF_COUNTRIES.match(name.strip()):
        return MULTIPLE_COUNTRIES
    return _geography_aliases().get(rule_key(name), name)


def load_entity_map() -> dict[str, tuple[str, str, str, str, str]]:
    """The entity map as {spelling key: (entity id, canonical name, type, ticker, map key)}; empty if not built yet."""
    if not ENTITY_MAP.exists():
        return {}
    with ENTITY_MAP.open() as f:
        return {r["spelling_key"]: (r["entity_id"], r["name"], r["label"], r["ticker"], r["key"]) for r in csv.DictReader(f)}


class CompanyFacts:
    """The extracted layer of one company, collected in memory before it is written.

    `nodes`: entity id -> (label, display name, extra properties).
    `edges`: (relation, subject id, object id, fiscal year) -> chunk_ids, evidence, verified,
    schema_version (the oldest schema version among the chunks that support the edge).
    `mentions`: (chunk id, entity id) pairs.
    `canonical`: node ids whose name comes from the entity map (written over any older name).
    """

    def __init__(self, ticker: str, filer_node_id: str, filer_name: str,
                 entity_map: dict | None = None, corpus_nodes: dict[str, str] | None = None):
        """Start an empty collection for one company.

        `filer_node_id` / `filer_name`: the company's structural node id and common name.
        `entity_map` (from `load_entity_map`) and `corpus_nodes` (ticker -> structural node id)
        turn on entity resolution; without them every spelling is its own node, which is what
        the resolver itself needs when it reads the raw names.
        """
        self.ticker = ticker
        self.filer_node_id = filer_node_id
        self.filer_name = filer_name.lower()
        self.entity_map = entity_map or {}
        self.corpus_nodes = corpus_nodes or {}
        self.nodes: dict[str, tuple[str, str, dict]] = {}
        self.canonical: set[str] = set()
        self.self_loops = 0   # facts dropped because subject and object are the same node
        self.edges: dict[tuple, dict] = defaultdict(
            lambda: {"chunk_ids": [], "evidence": [], "verified": False, "schema_version": None})
        self.mentions: set[tuple[str, str]] = set()

    def node(self, label: str, name: str, chunk_id: str, category: str | None = None) -> str:
        """Register an entity named in a chunk and return its node id.

        The filer maps to its own Company node. A Company/Product name found in the entity map
        becomes its entity: a corpus company's structural node (never rewritten here), or the
        entity's node with the canonical name and type. A name of a corpus company that the
        extraction typed as a Product (the app "Facebook", the database "MongoDB") gets its own
        product node (`corpus_product_id`), linked PRODUCT_OF the company by the entity map, so
        the company never points at itself. Places get their canonical name
        first (`geography_name`). Other names keep `entity_id`.
        """
        if label == "Company" and name.strip().lower() == self.filer_name:
            return self.filer_node_id
        if label == "Geography":
            name = geography_name(name)
        hit = self.entity_map.get(rule_key(name)) if label in RESOLVED_TYPES else None
        if hit and hit[3] and label == "Product":   # a product named like a corpus company ("Facebook")
            nid, display = corpus_product_id(hit[4]), tidy_name(name)
        elif hit and hit[3]:                         # a corpus company: its structural node
            nid = self.corpus_nodes[hit[3]]
            if nid != self.filer_node_id:
                self.mentions.add((chunk_id, nid))
            return nid
        elif hit:
            nid, display, label = hit[0], hit[1], hit[2]
            self.canonical.add(nid)
        else:
            nid, display = entity_id(label, name, self.ticker, category), tidy_name(name)
        if nid not in self.nodes:
            self.nodes[nid] = (label, display, {"category": category} if category else {})
        self.mentions.add((chunk_id, nid))
        return nid

    def edge(self, rel: str, s: str, o: str, fy: int, chunk_id: str, evidence: str, verified: bool,
             sv: int) -> None:
        """Add one chunk's support for a fact; facts repeated across chunks of a year share one edge.

        `sv` is the schema version the chunk's result was extracted under; if chunks of different
        versions share an edge, the oldest wins so a selective re-extraction still finds it.
        A fact whose subject and object are the same node ("Sun Microsystems ACQUIRED Sun
        Microsystems", an extraction slip) says nothing and is dropped and counted instead.
        """
        if s == o:
            self.self_loops += 1
            return
        e = self.edges[(rel, s, o, fy)]
        if chunk_id not in e["chunk_ids"]:
            e["chunk_ids"].append(chunk_id)
            e["evidence"].append(evidence)
        e["verified"] = e["verified"] or verified
        e["schema_version"] = sv if e["schema_version"] is None else min(e["schema_version"], sv)

    def add_risk(self, payload: dict, fy: int, chunk_id: str, sv: int) -> None:
        """Add a route-A result: the filer faces the risk, and the risk concerns the places in its title."""
        risk = self.node("Risk", payload["risk"] or payload["title"], chunk_id, payload["category"])
        self.edge("FACES_RISK", self.filer_node_id, risk, fy, chunk_id, payload["title"], True, sv)
        for place in payload.get("places") or []:
            if isinstance(place, str) and place.strip():
                geo = self.node("Geography", place, chunk_id)
                self.edge("CONCERNS", risk, geo, fy, chunk_id, payload["title"], True, sv)

    def add_facts(self, payload: dict, fy: int, chunk_id: str, sv: int) -> None:
        """Add a route-B result: every validated fact of one chunk."""
        for f in payload["facts"]:
            s = self.node(f["source_type"], f["source"], chunk_id, f.get("category") if f["source_type"] == "Risk" else None)
            o = self.node(f["target_type"], f["target"], chunk_id, f.get("category") if f["target_type"] == "Risk" else None)
            self.edge(f["relation"], s, o, fy, chunk_id, f["evidence"], bool(f.get("verified")), sv)


def collect(db: sqlite3.Connection, ticker: str, filer_node_id: str,
            entity_map: dict | None = None, corpus_nodes: dict[str, str] | None = None) -> CompanyFacts:
    """Read the company's results into a CompanyFacts: done items, and reuse items whose source is done.

    A reuse item whose source is waiting to be extracted again (e.g. after a requeue) is left
    out until the source is done, so an old result never reaches the graph. Pass `entity_map`
    and `corpus_nodes` to resolve names (the writer does; the resolver reads raw names).
    """
    cf = CompanyFacts(ticker, filer_node_id, filer_name(ticker), entity_map, corpus_nodes)
    rows = db.execute(
        "SELECT i.route, i.fiscal_year, i.chunk_id, r.payload, r.prompt_version FROM items i"
        " LEFT JOIN items s ON s.item_id = i.source"
        " JOIN results r ON r.item_id = COALESCE(i.source, i.item_id)"
        " WHERE i.ticker = ? AND ((i.kind = 'extract' AND i.status = 'done')"
        "  OR (i.kind = 'reuse' AND s.status = 'done'))"
        " ORDER BY i.fiscal_year, i.chunk_id", (ticker,))
    for route, fy, chunk_id, payload, prompt_version in rows:
        p = json.loads(payload)
        sv = SCHEMA_VERSION_BY_PROMPT[prompt_version]   # unknown prompt version: fail, don't guess
        if route == "A":
            cf.add_risk(p, fy, chunk_id, sv)
        else:
            cf.add_facts(p, fy, chunk_id, sv)
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
        by_label[label].append({"id": nid, "name": name, "props": props, "canonical": nid in cf.canonical})
    for label, rows in by_label.items():
        assert label in ENTITY_TYPES, label
        q(f"""UNWIND $rows AS row
              MERGE (e:__Node__ {{id: row.id}})
              SET e.name = CASE WHEN row.canonical OR e.name IS NULL THEN row.name ELSE e.name END
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
                      r.schema_version = row.schema_version
                  RETURN count(r) AS n""", param_map={"rows": rows, "t": t})[0]["n"]
        out[rel] += n
        out["edges_missing_node"] += len(rows) - n
    rows = [{"c": c, "e": e} for c, e in cf.mentions]
    n = q(f"""UNWIND $rows AS row
              MATCH (c:__Node__ {{id: row.c}}) MATCH (e:__Node__ {{id: row.e}})
              MERGE (c)-[r:{MENTIONS_RELATION} {{ticker: $t}}]->(e)
              RETURN count(r) AS n""", param_map={"rows": rows, "t": t})[0]["n"] if rows else 0
    out[MENTIONS_RELATION] = n
    out["mentions_missing_chunk"] = len(rows) - n
    out["deleted_orphans"] = delete_orphans(store)
    out["unverified"] = sum(not e["verified"] for e in cf.edges.values())
    out["self_loops_dropped"] = cf.self_loops
    return out


def delete_orphans(store) -> int:
    """Delete extracted entity nodes that no fact or mention uses any more (resolution edges alone don't count)."""
    return store.structured_query(f"""
        MATCH (e:__Entity__ {{origin: 'extraction'}})
        WHERE NOT EXISTS {{ MATCH (e)-[r]-() WHERE NOT type(r) IN {RESOLUTION_EDGES} }}
        DETACH DELETE e RETURN count(*) AS n""")[0]["n"]


def write_links(store, corpus_nodes: dict[str, str]) -> Counter:
    """Replace all entity-resolution links in Neo4j with those in `entity_links.csv`.

    A link is written only if both of its nodes exist (the entities are used by a loaded
    company); `corpus:TICKER` ids become the structural Company node. Returns counts.
    """
    q, out = store.structured_query, Counter()
    for rel in LINK_RELATIONS:
        out["deleted"] += q(f"MATCH ()-[r:{rel} {{origin: 'resolution'}}]->() DELETE r RETURN count(r) AS n")[0]["n"]
    if not ENTITY_LINKS.exists():
        return out
    node = lambda eid: corpus_nodes[eid.split(":", 1)[1]] if eid.startswith("corpus:") else eid
    by_rel = defaultdict(list)
    with ENTITY_LINKS.open() as f:
        for r in csv.DictReader(f):
            by_rel[r["relation"]].append({"c": node(r["child"]), "p": node(r["parent"]), "source": r["source"]})
    for rel, rows in by_rel.items():
        assert rel in LINK_RELATIONS, rel
        n = q(f"""UNWIND $rows AS row
                  MATCH (c:__Node__ {{id: row.c}}) MATCH (p:__Node__ {{id: row.p}})
                  MERGE (c)-[r:{rel} {{origin: 'resolution'}}]->(p) SET r.source = row.source
                  RETURN count(r) AS n""", param_map={"rows": rows})[0]["n"]
        out[rel] += n
        out["skipped_missing_node"] += len(rows) - n
    return out


def write_clusters(store) -> Counter:
    """Replace all RiskCluster / ThemeCluster nodes and IN_CLUSTER edges with those in `clusters.csv`.

    A cluster node is written only if at least one of its members is in the graph, and it
    keeps the cluster's name, category, size (number of wordings) and number of companies
    (over everything in the job store). Returns counts.
    """
    q, out = store.structured_query, Counter()
    for label in CLUSTER_LABELS:
        out["deleted_clusters"] += q(f"MATCH (c:{label}) DETACH DELETE c RETURN count(*) AS n")[0]["n"]
    if not CLUSTERS.exists():
        return out
    by_label = defaultdict(list)
    with CLUSTERS.open() as f:
        for r in csv.DictReader(f):
            by_label[f"{r['type']}Cluster"].append(
                {"node": r["node_id"], "id": r["cluster_id"], "name": r["cluster_name"],
                 "category": r["cluster_category"], "size": int(r["cluster_size"]), "filers": int(r["cluster_filers"])})
    for label, rows in by_label.items():
        assert label in CLUSTER_LABELS, label
        n = q(f"""UNWIND $rows AS row
                  MATCH (m:__Node__ {{id: row.node}})
                  MERGE (c:__Node__:{label} {{id: row.id}})
                  SET c.name = row.name, c.category = row.category, c.size = row.size,
                      c.n_filers = row.filers, c.origin = 'resolution'
                  MERGE (m)-[r:{CLUSTER_RELATION} {{origin: 'resolution'}}]->(c)
                  RETURN count(r) AS n""", param_map={"rows": rows})[0]["n"]
        out[f"{label} members"] += n
        out["members_not_in_graph"] += len(rows) - n
        out[f"{label} nodes"] = q(f"MATCH (c:{label}) RETURN count(c) AS n")[0]["n"]
    return out


def write(tickers: list[str] | None = None, db_path: Path = JOBS_DB) -> None:
    """Write the extracted layer of the given companies (default: all with results), then the entity links and clusters; print a summary."""
    if not db_path.exists():
        sys.exit(f"{db_path} does not exist: no extraction results yet")
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)   # read only: the runner may be writing
    if not tickers:
        tickers = [r[0] for r in db.execute(
            "SELECT DISTINCT i.ticker FROM items i JOIN results r USING (item_id) ORDER BY 1")]
    names = dict(zip(load_companies().ticker, load_companies().name))
    entity_map = load_entity_map()
    print(f"Entity map: {len(entity_map):,} spelling keys" if entity_map else "Entity map: none (names not resolved)")
    store = get_graph_store()
    ensure_indexes(store)
    for t in tickers:
        out = write_company(store, collect(db, t, names[t], entity_map, names))
        facts = {k: v for k, v in out.items() if k in RELATION_TYPES}
        print(f"{t}: {sum(facts.values())} fact edges {facts}, {out[MENTIONS_RELATION]} mentions, "
              f"{out['unverified']} unverified; missing nodes {out['edges_missing_node']}, "
              f"missing chunks {out['mentions_missing_chunk']}, orphans removed {out['deleted_orphans']}, self-loops dropped {out['self_loops_dropped']}")
    links = write_links(store, names)
    print(f"Entity links: {dict(links)}")
    print(f"Clusters: {dict(write_clusters(store))}")


if __name__ == "__main__":
    write(sys.argv[1:] or None)
