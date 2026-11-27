"""Stages 1-3 of entity resolution: collect the extracted names and settle what rules can settle.

What it does:
  - `name_inventory()` reads every extracted Company and Product name from the job store
    (through `graph.facts.collect`, so the names are exactly the ones the graph writer would
    use) and groups them by rule key (stage 1). Each group (`Key`) remembers its spellings,
    how often it was used as a Company or a Product, and which filers used it.
  - `rule_key()` (in `keys`) is the stage-1 rule: two names with the same key are the same thing.
    `merge_spacing()` adds a second one: keys that differ only by spaces are merged too, and
    `merge_versions()` a third: versions of one product ("PAN-OS 10.0", "PAN-OS 10.2") merge.
  - `match_aliases()` links keys to our 130 corpus companies through the alias table (stage 3).
  - `load_overrides()` reads the hand decisions in `overrides.csv` (stage 2). They use the
    same four decisions the LLM gives, so a hand decision simply takes the LLM's place.

Why it is needed: these steps cost nothing and settle most spellings (case, punctuation,
"Inc."). Only what they leave open goes on to candidate pairs and, later, to the LLM.
Risks and themes are not resolved here; they get a cluster layer of their own.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from sec10k.config import ENTITY_OVERRIDES, JOBS_DB, load_companies
from sec10k.extract.companies import load_aliases
from sec10k.graph.facts import collect
from sec10k.resolve.keys import RESOLVED_TYPES, rule_key

MAX_EVIDENCE = 3   # evidence quotes kept per key (context for the LLM judge)
# Pair decisions, shared by hand overrides and the LLM judge. "a PRODUCT_OF b": a is b's product.
DECISIONS = ("SAME", "PRODUCT_OF", "SUBSIDIARY_OF", "DIFFERENT")
@dataclass
class Key:
    """One rule key: every spelling that normalizes to it, over both types and all filers."""

    key: str
    variants: Counter = field(default_factory=Counter)   # spelling -> fact-edge ends using it
    by_label: Counter = field(default_factory=Counter)   # "Company" / "Product" -> fact-edge ends
    filers: set = field(default_factory=set)             # tickers whose facts use the key
    uses: Counter = field(default_factory=Counter)       # "COMPETES_WITH (object)" -> edge count
    evidence: list = field(default_factory=list)         # a few evidence quotes, for the LLM judge
    ticker: str | None = None                            # corpus company, if stage 3 matched it

    def absorb(self, other: "Key") -> None:
        """Take over another key's spellings, counts, filers, uses and evidence (when two keys merge)."""
        self.variants.update(other.variants)
        self.by_label.update(other.by_label)
        self.filers |= other.filers
        self.uses.update(other.uses)
        self.evidence = (self.evidence + other.evidence)[:MAX_EVIDENCE]

    @property
    def edges(self) -> int:
        """How many fact-edge ends use this key (how important it is)."""
        return sum(self.by_label.values())

    @property
    def label(self) -> str:
        """The type most edges use (a type clash is settled by majority; a tie counts as Company)."""
        return "Company" if self.by_label["Company"] >= self.by_label["Product"] else "Product"

    @property
    def name(self) -> str:
        """The display name: the most-used spelling ("Microsoft" over "Microsoft Corporation")."""
        return self.variants.most_common(1)[0][0]


def name_inventory(db_path: Path = JOBS_DB, tickers: list[str] | None = None) -> dict[str, Key]:
    """Every extracted Company/Product name in the job store, grouped by rule key (stage 1).

    Uses the graph writer's own `collect`, so a name here is a node there. The filer itself
    is not included (the writer already links it to its structural Company node).
    """
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)   # read only: a runner may be writing
    if not tickers:
        tickers = [r[0] for r in db.execute(
            "SELECT DISTINCT i.ticker FROM items i JOIN results r USING (item_id) ORDER BY 1")]
    filer_nodes = dict(zip(load_companies().ticker, load_companies().name))
    keys: dict[str, Key] = {}
    for t in tickers:
        cf = collect(db, t, filer_nodes[t])
        ends, uses, evidence = Counter(), defaultdict(Counter), defaultdict(list)
        for (rel, s, o, fy), e in cf.edges.items():
            for nid, role in ((s, "subject"), (o, "object")):
                ends[nid] += 1
                uses[nid][f"{rel} ({role})"] += 1
                if len(evidence[nid]) < MAX_EVIDENCE and e["evidence"][0] not in evidence[nid]:
                    evidence[nid].append(e["evidence"][0])
        for nid, (label, name, _) in cf.nodes.items():
            k = rule_key(name)
            if label not in RESOLVED_TYPES or not k:
                continue
            entry = keys.setdefault(k, Key(k))
            entry.variants[name] += ends[nid]
            entry.by_label[label] += ends[nid]
            entry.filers.add(t)
            entry.uses.update(uses[nid])
            entry.evidence = (entry.evidence + evidence[nid])[:MAX_EVIDENCE]
    merge_spacing(keys)
    merge_versions(keys)
    return keys


def merge_spacing(keys: dict[str, Key]) -> int:
    """Stage 1, second rule: merge keys that differ only by spaces ("lastpass" / "last pass"). Returns merges.

    A key that is a corpus-company alias always survives (so "nortonlifelock", a former name
    of GEN, keeps its link); otherwise the most-used key does. The survivor takes over the
    others' spellings, counts, filers, uses and evidence. Kept separate from `rule_key`
    because later stages compare keys word by word (prefix pairs).
    """
    aliases = alias_keys()
    groups: dict[str, list[str]] = {}
    for k in keys:
        groups.setdefault(k.replace(" ", ""), []).append(k)
    n = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        keep, *rest = sorted(members, key=lambda k: (k not in aliases, -keys[k].edges, k))
        for k in rest:
            keys[keep].absorb(keys.pop(k))
            n += 1
    return n


def alias_keys() -> dict[str, str]:
    """Rule key of every corpus-company alias -> its ticker."""
    return {rule_key(r.alias): r.ticker for r in load_aliases().itertuples()}


_VERSION = re.compile(r"\s+v?\d+(?:\.\d+)+[a-z]?$|\s+v\d+$", re.I)   # " 10.2", " v4.4", " 3.0a", " v2"


def version_base(name: str) -> str | None:
    """The rule key of a name without its trailing version number, or None if it has none.

    Only a dotted number ("10.2") or a "v" number ("v2") counts as a version, so names where a
    number is part of the product ("Office 365", "Windows 10", "Civil 3D") are left alone.
    """
    base = _VERSION.sub("", name.strip())
    return rule_key(base) if base != name.strip() and rule_key(base) else None


def merge_versions(keys: dict[str, Key]) -> int:
    """Stage 1, third rule: merge every version of a product into one key. Returns merges.

    All keys with the same version-free base ("pan os 10 0", "pan os 11 0") join the base key
    ("pan os"), which is created if no spelling without a version exists. The display name
    stays the most-used spelling. Run after `merge_spacing`.
    """
    n = 0
    for k in list(keys):
        bases = {version_base(v) for v in keys[k].variants} - {None}
        if len(bases) != 1 or (base := bases.pop()) == k:
            continue
        keys.setdefault(base, Key(base)).absorb(keys.pop(k))
        n += 1
    return n


def match_aliases(keys: dict[str, Key]) -> int:
    """Stage 3: set `ticker` on every key equal to the rule key of a corpus-company alias. Returns the count."""
    by_alias = alias_keys()
    n = 0
    for k, entry in keys.items():
        entry.ticker = by_alias.get(k)
        n += entry.ticker is not None
    return n


def load_overrides(path: Path = ENTITY_OVERRIDES) -> dict[tuple[str, str], tuple[str, str]]:
    """Stage 2: hand decisions as {(key a, key b): (decision, note)}. Names are given as written; keys are derived.

    Fails on an unknown decision, so a typo never silently turns into "no decision".
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    out = {}
    for r in df.itertuples():
        if r.decision not in DECISIONS:
            raise ValueError(f"{path.name}: unknown decision {r.decision!r} for {r.name_a} / {r.name_b}")
        out[(rule_key(r.name_a), rule_key(r.name_b))] = (r.decision, r.note)
    return out
