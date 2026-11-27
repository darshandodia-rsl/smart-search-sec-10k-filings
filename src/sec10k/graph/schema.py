"""The knowledge-graph schema: which kinds of things and links the LLM may extract from 10-K text.

What it does: lists the entity types (node labels), the relation types (edge labels), the
allowed (subject type, relation, object type) combinations, the fixed list of risk categories,
and the properties every extracted fact carries. Each entry has a short description, which
the extraction prompt shows to the LLM.

Why it is needed: the schema follows from the questions the system must answer (relationships
between companies, risks, products/segments/themes, geography, all over time). Keeping it in
one place means the extractor (Phase 3), the fact writer (Phase 4) and text-to-Cypher retrieval
(Phase 6) all agree on the same labels. Anything the LLM returns outside this schema is dropped.

Conventions:
  - Labels are CamelCase (`Company`), matching the Company nodes of the structural graph.
  - The filer is the subject of company facts: a 10-K is written from the filer's point of
    view ("we compete with Google" -> (Microsoft)-[:COMPETES_WITH]->(Google)).
  - Every fact edge is stored once per fiscal year (see FACT_PROPERTIES), so a fact that
    appears in 2016 and 2024 is two edges and each year keeps its own source.

Run: .venv/bin/python -m sec10k.graph.schema   (prints a summary and checks the schema)
"""

from __future__ import annotations

# --------------------------------------------------------------------------------------
# Entity types: node labels the LLM may create
# --------------------------------------------------------------------------------------
ENTITY_TYPES: dict[str, str] = {
    "Company": "A company or organization, e.g. 'Sony', 'TSMC'. Includes the filer itself.",
    "Product": "A named product, service, platform or brand sold by a company, e.g. 'Azure', 'Xbox'.",
    "Segment": "A business segment the company reports results for, e.g. 'Intelligent Cloud'.",
    "Risk": "A specific risk the company says it faces, named in a short phrase, e.g. "
            "'reliance on a limited number of chip manufacturers'. Must have a `category`.",
    "Theme": "A technology, trend or strategic focus, e.g. 'generative AI', 'cloud migration', "
             "'subscription model'.",
    "Geography": "A country or region, e.g. 'China', 'Taiwan', 'Europe'.",
}

# --------------------------------------------------------------------------------------
# Relation types: edge labels the LLM may create
# --------------------------------------------------------------------------------------
RELATION_TYPES: dict[str, str] = {
    "COMPETES_WITH": "The subject company names the object company as a competitor.",
    "HAS_SUPPLIER": "The object company provides something the subject itself uses: components, "
                    "manufacturing, or services such as data centers or cloud hosting for the "
                    "subject's own platform (e.g. 'we host our platform on AWS').",
    "INTEGRATES_WITH": "The subject company's products run on, are deployed on or integrate with "
                       "the object company's platform or products for customers (e.g. 'our firewall "
                       "can be deployed on VMware ESXi'). Not a supplier.",
    "HAS_CUSTOMER": "The object company is a customer of the subject.",
    "PARTNERS_WITH": "The two companies have a partnership, alliance or joint offering.",
    "ACQUIRED": "The subject company acquired (or agreed to acquire) the object company.",
    "OFFERS": "The company sells or provides the product or service.",
    "HAS_SEGMENT": "The company reports the business segment.",
    "INCLUDES": "The segment includes the product or service.",
    "FACES_RISK": "The company says it is exposed to the risk.",
    "CONCERNS": "The risk is tied to the country or region (e.g. export rules for China).",
    "FOCUSES_ON": "The company invests in, targets or emphasizes the theme.",
    "OPERATES_IN": "The company sells, manufactures or has significant operations in the place.",
    "LOCATED_IN": "The company is headquartered in the country or region.",
}

# --------------------------------------------------------------------------------------
# Allowed triples: (subject type, relation, object type). Anything else is rejected.
# --------------------------------------------------------------------------------------
VALIDATION_SCHEMA: list[tuple[str, str, str]] = [
    ("Company", "COMPETES_WITH", "Company"),
    ("Company", "HAS_SUPPLIER", "Company"),
    ("Company", "INTEGRATES_WITH", "Company"),
    ("Company", "HAS_CUSTOMER", "Company"),
    ("Company", "PARTNERS_WITH", "Company"),
    ("Company", "ACQUIRED", "Company"),
    ("Company", "OFFERS", "Product"),
    ("Company", "HAS_SEGMENT", "Segment"),
    ("Segment", "INCLUDES", "Product"),
    ("Company", "FACES_RISK", "Risk"),
    ("Risk", "CONCERNS", "Geography"),
    ("Company", "FOCUSES_ON", "Theme"),
    ("Company", "OPERATES_IN", "Geography"),
    ("Company", "LOCATED_IN", "Geography"),
]

# --------------------------------------------------------------------------------------
# Risk categories: the fixed list every Risk node must pick one from.
# The free-text Risk name keeps the detail; the category makes counting and comparing
# across companies and years reliable.
# --------------------------------------------------------------------------------------
RISK_CATEGORIES: dict[str, str] = {
    "competition": "Competitors, pricing pressure, market share, new entrants.",
    "cybersecurity": "Cyber attacks, data breaches, security vulnerabilities.",
    "data_privacy": "Privacy and data-protection laws and how data is handled.",
    "regulation_legal": "Other laws and regulation, antitrust, litigation, tax.",
    "geopolitical": "Trade policy, tariffs, export controls, sanctions, war, political instability.",
    "supply_chain": "Suppliers, manufacturers, component shortages, single sources.",
    "macroeconomic": "Economic conditions, inflation, recession, customer spending.",
    "financial": "Currency exchange, interest rates, debt, liquidity, credit, investments.",
    "business_disruption": "Outages, natural disasters, pandemics, climate events.",
    "technology_product": "Keeping up with technology change, product defects, AI and platform shifts.",
    "customers_demand": "Customer concentration, demand, sales cycles, renewals.",
    "intellectual_property": "Protecting own IP, infringement claims, open source.",
    "talent": "Hiring and keeping key people and skilled staff.",
    "growth_scaling": "Managing rapid growth, scaling operations, limited operating history, "
                      "keeping up past growth rates.",
    "brand_reputation": "Building and protecting the brand and reputation, public perception, "
                        "misuse of products harming reputation.",
    "strategic_transactions": "Acquisitions, divestitures, integration, partnerships.",
    "esg_climate": "Environmental, social and governance expectations, climate regulation.",
    "stock_governance": "Share price volatility, internal controls, anti-takeover provisions.",
    "other": "Anything that fits none of the above.",
}

# Properties the LLM fills in on entities (beyond the name).
ENTITY_PROPERTIES: dict[str, dict[str, str]] = {
    "Risk": {"category": "One of RISK_CATEGORIES."},
}

# --------------------------------------------------------------------------------------
# Properties set by the pipeline (not the LLM) on every extracted fact edge.
# The writer merges edges on (subject, relation, object, fiscal_year), so each year is its
# own edge with its own source; the source chunk lets an answer cite the exact passage and
# lets a re-extraction replace exactly the facts of one chunk.
# --------------------------------------------------------------------------------------
FACT_PROPERTIES: dict[str, str] = {
    "fiscal_year": "Fiscal year of the filing the fact comes from.",
    "ticker": "Ticker of the filing company.",
    "chunk_ids": "IDs of the chunks in that filing that state the fact.",
    "evidence": "The supporting quote from each chunk, in the same order as chunk_ids "
                "(for a risk: the risk-factor title).",
    "verified": "True if at least one chunk's quote and named entities were found in the chunk "
                "text; False marks a fact the model may have invented or reworded.",
    "schema_version": "SCHEMA_VERSION used when the fact was extracted.",
}

# Link from a Chunk to every entity its extracted facts name, so retrieval can go from a
# chunk found by vector search straight to its entities (LlamaIndex's own convention).
MENTIONS_RELATION = "MENTIONS"

# Raise this whenever the lists above change, so facts from an older schema can be found
# and re-extracted selectively instead of re-running the whole corpus.
SCHEMA_VERSION = 3

# The schema version each extraction prompt version was written for. The graph writer uses it
# to stamp every fact with the schema it was extracted under (from the prompt version stored
# with the result), not the schema that is current when the graph is written. Add an entry
# whenever PROMPT_VERSION is raised.
SCHEMA_VERSION_BY_PROMPT: dict[int, int] = {5: 2, 6: 3}


def check_schema() -> None:
    """Check that the schema is consistent; raise `ValueError` on the first problem.

    Every allowed triple must use known entity and relation types, and every relation type
    must be used by at least one triple. Catches typos before they silently drop facts.
    """
    used = set()
    for subj, rel, obj in VALIDATION_SCHEMA:
        for t in (subj, obj):
            if t not in ENTITY_TYPES:
                raise ValueError(f"unknown entity type {t!r} in triple {(subj, rel, obj)}")
        if rel not in RELATION_TYPES:
            raise ValueError(f"unknown relation type {rel!r} in triple {(subj, rel, obj)}")
        used.add(rel)
    if unused := set(RELATION_TYPES) - used:
        raise ValueError(f"relation types with no allowed triple: {sorted(unused)}")
    for label in ENTITY_PROPERTIES:
        if label not in ENTITY_TYPES:
            raise ValueError(f"properties defined for unknown entity type {label!r}")


if __name__ == "__main__":
    check_schema()
    print(f"Schema v{SCHEMA_VERSION}: {len(ENTITY_TYPES)} entity types, "
          f"{len(RELATION_TYPES)} relation types, {len(RISK_CATEGORIES)} risk categories\n")
    for subj, rel, obj in VALIDATION_SCHEMA:
        print(f"  ({subj})-[:{rel}]->({obj})")
