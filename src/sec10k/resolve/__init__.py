"""Entity resolution (Phase 5): decide which extracted names refer to the same real-world thing.

What it holds:
  - `keys`: `rule_key()`, the one name-normalization rule (also used by the graph writer).
  - `names`: the name inventory from the job store, rule keys (stage 1), hand overrides
    (stage 2) and matches to our 130 corpus companies (stage 3).
  - `candidates`: unclear pairs of names (stage 4) that need a judgement, with a report.
  - `judge`: the LLM judge for those pairs (stage 5), one saved decision per pair.
  - `entity_map`: joins everything into the entity map and the product/subsidiary links.
  - `overrides.csv`: hand decisions on pairs, in git; they always win.

Why it is needed: the LLM writes the same company many ways ("Cisco", "Cisco Systems, Inc."),
and the graph writer makes one node per spelling, so facts about one company are scattered.
This package builds the map from every spelling to one entity; the graph writer (`graph.facts`)
uses it. It reads only the job store, so it runs without Neo4j and re-runs as filings arrive.
"""
