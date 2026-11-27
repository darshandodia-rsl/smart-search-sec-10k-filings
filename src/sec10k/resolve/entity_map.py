"""Build the entity map: every extracted Company/Product spelling -> one real-world entity.

What it does (no LLM, no Neo4j):
  1. Takes the keys from stages 1-3 (`names`: rules, alias table) and the pair decisions:
     hand overrides (`overrides.csv`) and the LLM judge's answers (`decisions.csv`). For a pair
     with both, the override wins. Decisions about keys that no longer exist (because a rule
     has since merged them) are counted as stale and ignored.
  2. Joins keys into entities: keys of one corpus company (aliases such as "google" and
     "alphabet"), then along SAME decisions, following chains (A = B and B = C makes
     one entity), with a union-find (a structure that tracks which group each item is in).
  3. Gives each entity an id, a name and a type:
       - an entity holding a corpus company (alias match) is that company: id "corpus:TICKER",
         name and ticker from the alias table, type Company;
       - otherwise id "<Type>:<key>", name = its most-used spelling, type = the type most of its
         edges use, unless a decision says what it is (the owned side of PRODUCT_OF is a
         Product; both sides of SUBSIDIARY_OF and the owner in PRODUCT_OF are Companies).
  4. Keeps PRODUCT_OF / SUBSIDIARY_OF as links between entities, and adds one PRODUCT_OF
     link (source "rule") for every corpus-company name the extraction also used as a Product
     (the app "Facebook" -> Meta): the graph writer gives those mentions their own product node.
  5. Reports conflicts instead of guessing: a DIFFERENT pair that a SAME chain joined anyway,
     two corpus companies in one entity, a link inside one entity, and a product that is also
     an owner. Each needs a hand decision in `overrides.csv`.
  Writes `data/resolution/entity_map.csv` (one row per spelling key: entity id, name, type,
  ticker, method) and `data/resolution/entity_links.csv` (child, relation, parent, source).

Why it is needed: this is the output of entity resolution. The graph writer looks each
extracted name up here (by its rule key), so all spellings of one thing become one node, and
the links become PRODUCT_OF / SUBSIDIARY_OF edges. Everything is rebuilt from its inputs on
every run, so it re-runs cleanly as new filings, overrides or decisions arrive.

Run: .venv/bin/python -m sec10k.resolve.entity_map
"""

from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from sec10k.config import ENTITY_LINKS, ENTITY_MAP, RESOLUTION_DIR
from sec10k.extract.llm_extract import filer_name
from sec10k.resolve.judge import load_decisions
from sec10k.resolve.keys import corpus_product_id, rule_key
from sec10k.resolve.names import Key, load_overrides, match_aliases, name_inventory

LINK_DECISIONS = ("PRODUCT_OF", "SUBSIDIARY_OF")


class UnionFind:
    """Groups items that are joined, directly or through a chain (A-B, B-C puts A, B, C in one group)."""

    def __init__(self, items):
        """Start with every item in its own group."""
        self.parent = {x: x for x in items}

    def find(self, x):
        """The representative of x's group (with path halving, so repeated lookups stay fast)."""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b) -> None:
        """Put a's and b's groups together."""
        self.parent[self.find(a)] = self.find(b)


def pair_decisions(keys: dict[str, Key]) -> tuple[list[tuple[str, str, str, str]], int]:
    """All usable decisions as (a, b, decision, source), overrides beating the LLM; plus the stale count.

    For PRODUCT_OF / SUBSIDIARY_OF, a is the owned side. A decision is stale when one of its
    keys is no longer a key (a rule merged it into another) or both keys are now the same.
    """
    merged: dict[frozenset, tuple[str, str, str, str]] = {}
    d = load_decisions()
    for r in d.itertuples():
        merged[frozenset((r.a, r.b))] = (r.a, r.b, r.decision, "llm")
    for (a, b), (decision, _) in load_overrides().items():
        merged[frozenset((a, b))] = (a, b, decision, "override")
    usable = [x for x in merged.values() if x[0] in keys and x[1] in keys and x[0] != x[1]]
    return usable, len(merged) - len(usable)


def build(keys: dict[str, Key]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Join keys into entities; return (map rows, link rows, report numbers and conflicts)."""
    decisions, stale = pair_decisions(keys)
    uf = UnionFind(keys)
    by_ticker: dict[str, str] = {}
    for k, entry in keys.items():   # aliases of one corpus company ("google", "alphabet") are one entity
        if entry.ticker:
            uf.union(k, by_ticker.setdefault(entry.ticker, k))
    joined_by: dict[str, str] = {}
    for a, b, decision, source in decisions:
        if decision == "SAME":
            uf.union(a, b)
            joined_by.setdefault(a, source)
            joined_by.setdefault(b, source)
    groups = defaultdict(list)
    for k in keys:
        groups[uf.find(k)].append(k)

    conflicts = []
    # Types forced by link decisions, per group.
    forced = defaultdict(set)
    for a, b, decision, source in decisions:
        if decision == "PRODUCT_OF":
            forced[uf.find(a)].add("Product")
            forced[uf.find(b)].add("Company")
        elif decision == "SUBSIDIARY_OF":
            forced[uf.find(a)].add("Company")
            forced[uf.find(b)].add("Company")

    entity_of: dict[str, dict] = {}   # group root -> entity record
    for root, members in groups.items():
        tickers = sorted({keys[k].ticker for k in members if keys[k].ticker})
        if len(tickers) > 1:
            conflicts.append(f"two corpus companies joined by SAME: {tickers} via {sorted(members)[:6]}")
        labels = Counter()
        variants = Counter()
        for k in members:
            labels.update(keys[k].by_label)
            variants.update(keys[k].variants)
        if tickers:
            ticker, label, name = tickers[0], "Company", filer_name(tickers[0])
            eid = f"corpus:{ticker}"
        else:
            ticker = ""
            if len(forced[root]) > 1:
                conflicts.append(f"both a product and an owner: {variants.most_common(1)[0][0]}")
            label = (next(iter(forced[root])) if len(forced[root]) == 1
                     else "Company" if labels["Company"] >= labels["Product"] else "Product")
            name = variants.most_common(1)[0][0]
            eid = f"{label}:{rule_key(name)}"
        entity_of[root] = {"entity_id": eid, "name": name, "label": label, "ticker": ticker,
                           "edges": sum(labels.values()), "keys": len(members), "spellings": len(variants)}

    # Two groups may produce the same entity id (e.g. two unjoined keys with the same best
    # spelling); that is a merge nobody decided, so report it.
    by_id = Counter(e["entity_id"] for e in entity_of.values())
    conflicts += [f"entity id used by {n} unjoined groups: {i}" for i, n in by_id.items() if n > 1]

    for a, b, decision, source in decisions:
        if decision == "DIFFERENT" and uf.find(a) == uf.find(b):
            conflicts.append(f"DIFFERENT ({source}) but joined by a SAME chain: {keys[a].name} / {keys[b].name}")

    links, seen = [], set()
    for a, b, decision, source in decisions:
        if decision not in LINK_DECISIONS:
            continue
        child, parent = entity_of[uf.find(a)], entity_of[uf.find(b)]
        if child is parent:
            conflicts.append(f"{decision} inside one entity: {keys[a].name} / {keys[b].name}")
            continue
        if (child["entity_id"], decision, parent["entity_id"]) in seen:
            continue
        seen.add((child["entity_id"], decision, parent["entity_id"]))
        links.append({"child": child["entity_id"], "relation": decision, "parent": parent["entity_id"],
                      "child_name": child["name"], "parent_name": parent["name"], "source": source})

    for k, entry in keys.items():   # products named like their corpus company
        e = entity_of[uf.find(k)]
        if e["ticker"] and entry.by_label["Product"]:
            links.append({"child": corpus_product_id(k), "relation": "PRODUCT_OF", "parent": e["entity_id"],
                          "child_name": entry.name, "parent_name": e["name"], "source": "rule"})

    rows = []
    for k, entry in keys.items():
        e = entity_of[uf.find(k)]
        method = "alias" if entry.ticker else joined_by.get(k, "rule")
        spelling_keys = {rule_key(v) for v in entry.variants} | {k}
        for sk in sorted(spelling_keys):
            rows.append({"spelling_key": sk, "key": k, "method": method if sk == k else "rule"} |
                        {c: e[c] for c in ("entity_id", "name", "label", "ticker")})
    report = {"stale": stale, "decisions": Counter((d[2], d[3]) for d in decisions),
              "conflicts": conflicts, "entities": entity_of}
    return pd.DataFrame(rows), pd.DataFrame(links), report


def main() -> None:
    """Build the map from the current inventory and decisions, write both files and print a report."""
    keys = name_inventory()
    match_aliases(keys)
    rows, links, rep = build(keys)
    RESOLUTION_DIR.mkdir(parents=True, exist_ok=True)
    dup = rows.spelling_key.duplicated(keep=False)
    if dup.any():   # one spelling key must lead to one entity, or lookups would be ambiguous
        raise SystemExit(f"spelling keys mapped twice: {rows[dup].spelling_key.unique()[:10]}")
    rows.to_csv(ENTITY_MAP, index=False)
    links.to_csv(ENTITY_LINKS, index=False)

    ents = pd.DataFrame(rep["entities"].values())
    print(f"Keys: {len(keys):,} -> entities: {len(ents):,} "
          f"({(ents.label == 'Company').sum():,} Company, {(ents.label == 'Product').sum():,} Product); "
          f"{(ents.ticker != '').sum()} are corpus companies")
    print(f"Decisions used: {dict(rep['decisions'])}; stale (keys merged by rules since): {rep['stale']}")
    print(f"Links: {links.relation.value_counts().to_dict() if len(links) else {}}")
    print(f"Spelling keys in the map: {len(rows):,}; method: {rows.method.value_counts().to_dict()}")
    print("\nBiggest entities (by keys joined):")
    for e in ents.sort_values(["keys", "edges"], ascending=False).head(12).itertuples():
        print(f"  {e.entity_id:28} {e.name[:30]:30} {e.label:8} keys {e.keys}, spellings {e.spellings}, edges {e.edges}")
    print(f"\nConflicts ({len(rep['conflicts'])}): need a hand decision in overrides.csv")
    for c in rep["conflicts"]:
        print("  " + c)
    print(f"\nWritten: {ENTITY_MAP}, {ENTITY_LINKS}")


if __name__ == "__main__":
    main()
