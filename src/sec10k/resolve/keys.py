"""The name key used everywhere in entity resolution: `rule_key()`, plus the types it resolves.

What it does: turns a name into its normalized form ("Cisco Systems, Inc." -> "cisco
systems"). Two names with the same key count as the same thing.

Why it is needed: the resolver builds the entity map from these keys, and the graph writer
(`graph.facts`) looks names up in that map by the same key. Both must use exactly one rule,
so it lives here, in a module that imports nothing else from the project (the graph writer
can use it without pulling in the resolver, which itself reads names through the writer).
"""

from __future__ import annotations

import re

RESOLVED_TYPES = ("Company", "Product")   # extracted entity types that entity resolution maps
# Legal-form words dropped from the end of a name ("SAP SE", "Cisco Systems, Inc.").
LEGAL_SUFFIXES = {"inc", "incorporated", "corp", "corporation", "co", "company", "plc", "ltd",
                  "limited", "llc", "lp", "llp", "gmbh", "ag", "sa", "se", "nv", "bv", "ab", "holdings"}


def rule_key(name: str) -> str:
    """The normalized form of a name: lower case, '&' as 'and', punctuation removed, legal suffixes dropped.

    "Cisco Systems, Inc." and "Cisco Systems Inc" both become "cisco systems"; "Panorama™"
    becomes "panorama". A part in brackets is dropped ("International Business Machines (IBM)"
    -> "international business machines"), because it repeats the name as an abbreviation.
    """
    s = re.sub(r"\([^)]*\)", " ", name.lower()).replace("&", " and ")
    words = re.sub(r"[^a-z0-9]+", " ", s).split()
    while len(words) > 1 and words[-1] in LEGAL_SUFFIXES:
        words.pop()
    return " ".join(words)


def corpus_product_id(key: str) -> str:
    """Node id of a product named like a corpus company, e.g. the app "Facebook" of Meta: "Product:facebook".

    `key` is the entity-map key of the name. The graph writer gives such Product-typed
    mentions this node (not the company's node) and the entity map links it PRODUCT_OF the
    company; both build the id here so they always agree.
    """
    return f"Product:{key}"
