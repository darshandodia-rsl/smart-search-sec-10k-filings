"""Survey the extracted Company and Product names in Neo4j, before building entity resolution.

What it does (read-only; it never writes to Neo4j):
  1. Reads every extracted Company and Product node with its number of fact edges and the
     filers whose facts use it.
  2. Rule groups: names that become identical under `resolve.names.rule_key` (lower case,
     no punctuation, no legal suffixes like "Inc." or "Corporation"). These merge by rules alone.
  3. Corpus matches: rule groups whose key equals an alias of one of our 130 companies
     (`company_aliases.csv`). These should join that company's structural Company node.
  4. Type clashes: rule keys that exist both as a Company and as a Product node.
  5. Family groups: rule groups that share their first word ("Amazon", "Amazon Web Services",
     "Amazon Prime"). These are the unclear cases: same company, subsidiary, or product?
  6. Spot-check: a fixed random sample of fact edges (with evidence) from the development slice.
  Writes all rule groups to `data/profile/entity_name_groups.csv` for review.

Why it is needed: it shows how much of Phase 5 entity resolution simple rules can do, which
cases are left for embeddings and the LLM, and gives the hand spot-check of the development
slice. Its numbers decide how the resolution step is built.

Run: .venv/bin/python -m sec10k.analysis.entity_names
"""

from __future__ import annotations

import random
from collections import defaultdict

import pandas as pd

from sec10k.config import PROFILE_DIR
from sec10k.graph.store import get_graph_store
from sec10k.resolve.names import alias_keys, rule_key

DEV_SLICE = ["PANW", "CRWD", "TENB", "RPD", "QLYS", "FTNT", "ZS", "OKTA", "NET", "S", "VRNS",
             "MSFT", "GOOGL", "GEN", "CRM", "ORCL", "NOW"]
# Words too common to define a family on their own ("The Home Depot", "Cloud Software Group").
GENERIC_FIRST = {"the", "cloud", "global", "national", "american", "united", "international",
                 "first", "general", "digital", "data", "open", "new", "other", "us", "u"}


def load_nodes(store) -> pd.DataFrame:
    """Every extracted Company/Product node: id, name, label, fact-edge count, filers using it."""
    rows = store.structured_query("""
        MATCH (e:__Entity__) WHERE e.origin = 'extraction' AND (e:Company OR e:Product)
        OPTIONAL MATCH (e)-[r]-() WHERE type(r) <> 'MENTIONS'
        RETURN e.id AS id, e.name AS name,
               CASE WHEN e:Company THEN 'Company' ELSE 'Product' END AS label,
               count(r) AS edges, collect(DISTINCT r.ticker) AS filers""")
    df = pd.DataFrame(rows)
    df["key"] = df.name.map(rule_key)
    return df


def print_groups(title: str, groups: pd.DataFrame, n: int) -> None:
    """Print the `n` biggest groups (by fact edges) as 'key: name (edges), ...'."""
    print(f"\n{title}")
    for key, g in groups.head(n).iterrows():
        names = ", ".join(f"{r.name} ({r.edges})" for r in g.members)
        print(f"  {key}: {names}")


def survey() -> None:
    """Run all survey sections and print the report (see the module docstring)."""
    store = get_graph_store()
    df = load_nodes(store)
    aliases = alias_keys()
    print(f"Extracted nodes: {len(df):,} ({(df.label == 'Company').sum():,} Company, "
          f"{(df.label == 'Product').sum():,} Product), {df.edges.sum():,} fact-edge ends")

    # 2. Rule groups (per label).
    groups = (df.groupby(["label", "key"])
                .apply(lambda g: pd.Series({"n": len(g), "edges": g.edges.sum(),
                                            "members": list(g.sort_values("edges", ascending=False).itertuples()),
                                            "filers": len(set().union(*g.filers))}), include_groups=False)
                .sort_values("edges", ascending=False))
    for label in ["Company", "Product"]:
        lab = groups.loc[label]
        multi = lab[lab.n > 1]
        print(f"\n{label}: {len(df[df.label == label]):,} nodes -> {len(lab):,} after rules "
              f"({multi.n.sum() - len(multi):,} merged away, in {len(multi):,} groups)")
        print_groups(f"  Biggest {label} rule groups:", multi, 8)

    # 3. Corpus matches.
    comp = groups.loc["Company"].copy()
    comp["ticker"] = comp.index.map(aliases.get)
    matched = comp[comp.ticker.notna()]
    print(f"\nCorpus matches (Company rule key = an alias key): {len(matched):,} groups, "
          f"{matched.n.sum():,} nodes, {matched.edges.sum():,} of {comp.edges.sum():,} Company "
          f"edge ends ({matched.edges.sum() / comp.edges.sum():.0%}), "
          f"{matched.ticker.nunique()} of 130 companies")
    print("  Most-used: " + ", ".join(f"{k} -> {t} ({e})" for k, t, e in
                                       matched.sort_values("edges", ascending=False)[["ticker", "edges"]]
                                       .head(15).itertuples()))

    # 4. Type clashes.
    clash = sorted(set(groups.loc["Company"].index) & set(groups.loc["Product"].index),
                   key=lambda k: -(groups.loc[("Company", k), "edges"] + groups.loc[("Product", k), "edges"]))
    print(f"\nType clashes (same key as Company and as Product): {len(clash):,}")
    print("  " + ", ".join(f"{k} (C {groups.loc[('Company', k), 'edges']} / P {groups.loc[('Product', k), 'edges']})"
                           for k in clash[:20]))

    # 5. Family groups: Company rule groups sharing a non-generic first word.
    fam = defaultdict(list)
    for key, g in comp.iterrows():
        first = key.split()[0] if key else ""
        if first and first not in GENERIC_FIRST and len(first) > 2:
            fam[first].append((key, g.edges, g.ticker if isinstance(g.ticker, str) else ""))
    families = sorted(((f, m) for f, m in fam.items() if len(m) > 1), key=lambda x: -sum(e for _, e, _ in x[1]))
    print(f"\nFamily groups (Company keys sharing a first word, i.e. unclear cases): {len(families):,} "
          f"families, {sum(len(m) for _, m in families):,} keys")
    for first, members in families[:15]:
        print(f"  {first}: " + ", ".join(f"{k}{' [' + t + ']' if t else ''} ({e})"
                                         for k, e, t in sorted(members, key=lambda x: -x[1])[:8]))

    # 6. Spot-check sample of the development slice.
    facts = store.structured_query("""
        MATCH (s)-[r]->(o) WHERE r.ticker IN $t AND type(r) <> 'MENTIONS' AND type(r) <> 'FACES_RISK'
              AND type(r) <> 'CONCERNS'
        RETURN r.ticker AS t, r.fiscal_year AS fy, s.name AS s, type(r) AS rel, o.name AS o,
               r.verified AS v, r.evidence[0] AS ev""", param_map={"t": DEV_SLICE})
    print(f"\nSpot-check: 15 of {len(facts):,} dev-slice relation edges (seed 5)")
    for f in random.Random(5).sample(facts, 15):
        print(f"  {f['t']} FY{f['fy']}: {f['s']} -{f['rel']}-> {f['o']}{'' if f['v'] else '  [unverified]'}\n"
              f"      \"{(f['ev'] or '')[:160]}\"")

    out = PROFILE_DIR / "entity_name_groups.csv"
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    flat = groups.reset_index()
    flat["names"] = flat.members.map(lambda m: " | ".join(f"{r.name} ({r.edges})" for r in m))
    flat["ticker"] = [aliases.get(k) if lab == "Company" else None for lab, k in zip(flat.label, flat.key)]
    flat[["label", "key", "n", "edges", "filers", "ticker", "names"]].to_csv(out, index=False)
    print(f"\nAll rule groups written to {out}")


if __name__ == "__main__":
    survey()
