"""Tier 4 extraction: schema-guided LLM extraction for routes A and B (Phase 3 prototype).

What it does:
  - Route A: sends a filing's distinct Item 1A risk titles to the LLM, 50 per call (25 on Groq), and gets
    back for each title a short risk name, one risk category and the places it names.
  - Route B: sends the filing's route-B chunks (new, and naming a company other than the
    filer per tier 2 or tier 3), 8 per call (4 on Groq), and gets back facts that fit the schema in
    `graph/schema.py`, each with an evidence quote of at most 10 words. Facts come in a
    compact form, ["SELF", "COMPETES_WITH", "Company:Oracle", "evidence"], because output
    tokens are the largest part of the cost: types sit in front of names (no separate
    entity list) and "SELF" stands for the filing company.
  - Validates every answer in code: malformed facts, unknown types, triples not in
    VALIDATION_SCHEMA and unknown risk categories are dropped and counted; a kept fact is
    "verified" when the entities it names occur in the chunk and its evidence quote matches
    the chunk (exactly, in "..."-joined pieces, or by most of its words); others are flagged. Prompt problems show up as numbers.
  - Caches every LLM answer on disk under a hash of (prompt version, backend, model, prompt),
    so re-running never pays for the same call twice.
  - Writes results to `data/extract_proto/{accession}_{A|B}.json`.

Why it is needed: this is the step that turns text into graph facts. The prototype checks
the prompts on a few filings by hand and measures real tokens per call, before the
production pipeline (Phase 4) runs it on the whole corpus.

The prompts are built from the schema, so changing the schema changes the prompts. Raise
PROMPT_VERSION whenever a prompt changes; cached answers of older versions are then ignored.

Run: .venv/bin/python -m sec10k.extract.llm_extract TICKER FISCAL_YEAR [--backend claude_cli]
       [--max-a-calls N] [--max-b-calls N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass

from sec10k.config import EXTRACT_PROTO_DIR, LLM_CACHE_DIR, load_manifest
from sec10k.extract.companies import CompanyMatcher, load_aliases
from sec10k.extract.entity_cleaning import load_entities, names_other_company
from sec10k.extract.patterns import find_cues
from sec10k.graph.schema import (ENTITY_TYPES, RELATION_TYPES, RISK_CATEGORIES, SCHEMA_VERSION,
                                 VALIDATION_SCHEMA)
from sec10k.ingest.chunker import Chunk, iter_chunks
from sec10k.llm.providers import LLMProvider, LLMResponse, get_llm

PROMPT_VERSION = 2
# Batch sizes per call: (risk titles, route-B chunks, max output tokens). Groq's free tier
# allows 8K tokens per minute, counted as input + allowed output (which includes gpt-oss's
# reasoning), so one Groq call must stay well under 8K; the other backends take larger batches.
BATCH_DEFAULT = (50, 8, 8000)
BATCH_BY_BACKEND = {"groq": (25, 4, 4000)}
SECTION_NAMES = {"item_1": "Item 1 Business", "item_1A": "Item 1A Risk Factors",
                 "item_7": "Item 7 MD&A", "item_7A": "Item 7A Market Risk"}

# --------------------------------------------------------------------------------------
# Prompts (built from the schema)
# --------------------------------------------------------------------------------------
_CATEGORIES = "\n".join(f"- {k}: {v}" for k, v in RISK_CATEGORIES.items())

SYSTEM_A = f"""You classify risk factors from US companies' annual reports (10-K).
For each numbered risk-factor title, return:
- "risk": a short, generic name of the risk in 3-10 words, without the company's name
  (e.g. "reliance on a limited number of chip manufacturers")
- "category": exactly one of the categories below
- "places": the countries or regions the title itself names (empty list if none)

Categories:
{_CATEGORIES}

Answer with JSON only, no other text:
{{"risks": [{{"id": 1, "risk": "...", "category": "...", "places": []}}, ...]}}
Return exactly one item per title, using the title's number as "id"."""

_ENTITIES = "\n".join(f"- {k}: {v}" for k, v in ENTITY_TYPES.items())
_RELATIONS = "\n".join(f"- {k}: {v}" for k, v in RELATION_TYPES.items())
_TRIPLES = "\n".join(f"- ({s})-[:{r}]->({o})" for s, r, o in VALIDATION_SCHEMA)
_CATEGORY_NAMES = ", ".join(RISK_CATEGORIES)

SYSTEM_B = f"""You extract knowledge-graph facts from passages of a US company's annual report (10-K).

Entity types:
{_ENTITIES}

Relation types:
{_RELATIONS}

Allowed facts (source type, relation, target type); use no others:
{_TRIPLES}

Risk categories: {_CATEGORY_NAMES}

Rules:
1. Extract only facts the passage states. Use no outside knowledge.
2. The report is written by the filing company ("we", "our", "the Company"). Write it as SELF.
   "we compete with Oracle" -> ["SELF", "COMPETES_WITH", "Company:Oracle", ...].
3. Write every other entity as "Type:Name", e.g. "Company:Oracle", "Product:Azure",
   "Segment:Intelligent Cloud", "Geography:China", "Theme:generative AI".
   Write a risk as "Risk:category:short name", e.g. "Risk:supply_chain:reliance on a few chip makers".
4. Name things as the passage does, using the fullest form it gives ("Amazon Web Services",
   not "AWS", if both appear).
5. A business the filing company owns (a subsidiary or acquired company, e.g. LinkedIn) is a
   Company, not a Product. Use ACQUIRED only when the passage says it was acquired.
6. Generic groups ("competitors", "our customers", "large banks") are not entities.
7. The 4th element is evidence: the shortest exact words from the passage that state the
   fact, at most 10 words.
8. If a passage has no fact that fits, give it an empty list.

Answer with compact JSON only, no other text: an object mapping each passage number to its facts.
{{"1": [["SELF", "COMPETES_WITH", "Company:Oracle", "faces competition from Amazon, Google, IBM, Oracle"],
       ["Segment:Intelligent Cloud", "INCLUDES", "Product:Azure", "including Azure and other cloud services"]],
 "2": []}}"""


def _filer_name(ticker: str) -> str:
    """The company's common name (first alias in the alias table), e.g. "Microsoft"."""
    return load_aliases().query("ticker == @ticker").alias.iloc[0]


def prompt_a(name: str, ticker: str, fy: int, titles: list[str]) -> str:
    """The user prompt of one route-A call: the company and its numbered risk titles."""
    lines = "\n".join(f"{i}. {t}" for i, t in enumerate(titles, 1))
    return f"Company: {name} ({ticker}), 10-K fiscal year {fy}.\n\nRisk-factor titles:\n{lines}"


def prompt_b(name: str, ticker: str, fy: int, chunks: list[Chunk]) -> str:
    """The user prompt of one route-B call: the filing company and its numbered passages."""
    parts = [f"Filing company: {name} ({ticker}), 10-K fiscal year {fy}.\n"]
    for i, c in enumerate(chunks, 1):
        head = f" | heading: {c.heading}" if c.heading else ""
        parts.append(f"--- Passage {i} ({SECTION_NAMES[c.section]}{head})\n{c.text}\n")
    return "\n".join(parts)


# --------------------------------------------------------------------------------------
# Calling the LLM with a cache
# --------------------------------------------------------------------------------------
@dataclass
class CallResult:
    """A parsed LLM answer plus its token use and whether it came from the cache."""

    data: dict | None     # parsed JSON, or None if the answer was not valid JSON
    response: LLMResponse
    cached: bool
    seconds: float


def _parse_json(text: str) -> dict | None:
    """Parse the model's JSON answer, tolerating a ```json fence around it."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return None


def call_cached(llm: LLMProvider, system: str, prompt: str, max_tokens: int) -> CallResult:
    """Return the LLM's answer to (system, prompt), from the disk cache if it was asked before.

    The cache key includes PROMPT_VERSION, the backend and the model, so changing any of them
    asks again. Cached answers keep their original token counts for reporting.
    """
    key = hashlib.sha1(json.dumps([PROMPT_VERSION, llm.backend, llm.model, system, prompt]).encode()).hexdigest()
    path = LLM_CACHE_DIR / f"{key}.json"
    if path.exists():
        d = json.loads(path.read_text())
        r = LLMResponse(**d["response"])
        return CallResult(_parse_json(r.text), r, True, d["seconds"])
    t = time.time()
    r = llm.complete(prompt, system=system, max_tokens=max_tokens)
    secs = time.time() - t
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"response": r.__dict__, "seconds": secs, "prompt_version": PROMPT_VERSION}))
    tmp.replace(path)
    return CallResult(_parse_json(r.text), r, False, secs)


# --------------------------------------------------------------------------------------
# Choosing the inputs of a filing
# --------------------------------------------------------------------------------------
def filing_inputs(ticker: str, fy: int) -> tuple[dict, list[str], list[Chunk]]:
    """Return (filing row, route-A titles, route-B chunks) for one filing.

    Route-A titles are the filing's distinct Item 1A headings not already in an earlier
    filing of the company; all-caps section headings are left out. Route-B chunks are chunks whose text is new for the company and
    that name another company (tier 2 cue/corpus name or tier 3 cleaned GLiNER entity).
    """
    m = load_manifest()
    row = m[(m.ticker == ticker) & (m.fiscal_year == fy)].iloc[0]
    seen_hashes, seen_titles, chunks = set(), set(), []
    for c in iter_chunks([ticker]):
        if c.fiscal_year < fy:
            seen_hashes.add(c.content_hash)
            if c.section == "item_1A" and c.heading:
                seen_titles.add(c.heading)
        elif c.fiscal_year == fy:
            chunks.append(c)
    titles = list(dict.fromkeys(c.heading for c in chunks if c.section == "item_1A" and c.heading
                                and c.heading not in seen_titles
                                and not c.heading.isupper()))  # skip section headings ("OPERATIONAL RISKS")
    ents = load_entities(row.accession_number)
    matcher = CompanyMatcher()
    route_b = [c for c in chunks if c.content_hash not in seen_hashes and (
        names_other_company(ents.get(c.chunk_id, []))
        or any(q.named for q in find_cues(c.text))
        or any(not x.is_self for x in matcher.find(c.text, ticker)))]
    return row, titles, route_b


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------
# Share of a quote's words that must occur in the passage for a reworded quote to count.
WORD_SHARE = 0.7
# Node types whose names are taken from the passage, so the name itself must occur in it.
# Risk and Theme names are short summaries written by the LLM and are not checked.
_NAMED_TYPES = {"Company", "Product", "Segment", "Geography"}


def _squash(text: str) -> str:
    """Lowercase, unify quotes, dashes and spaces, drop trademark signs and collapse whitespace.

    Used before every text comparison, so typographic differences (curly quotes, non-breaking
    hyphens, "™") do not make a quote that is in the passage look missing.
    """
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"(?<=[A-Za-z0-9])TM(?=\b|\d)", "", text)  # "EyeQTM5": a ™ sign turned into letters
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    text = re.sub(r"[\u00a0\u202f]", " ", text)
    text = re.sub(r"[™®©]", "", text)
    return re.sub(r"\s+", " ", text.lower()).strip()


_SUFFIX = re.compile(r",?\s+(inc|corp|corporation|co|ltd|limited|llc|plc|n\.?v|s\.?a|ag|se)\.?$")


def _name_in(name: str, passage: str) -> bool:
    """True if an entity name occurs in the (squashed) passage.

    Also accepts the name without a company suffix ("STMicroelectronics N.V." ->
    "STMicroelectronics") and its initials as a word ("Amazon Web Services" -> "AWS"), since
    the prompt asks for the fullest name, which may be spelled out elsewhere in the filing.
    """
    n = _SUFFIX.sub("", _squash(name))
    if n in passage:
        return True
    words = re.findall(r"[a-z0-9]+", n)
    initials = "".join(w[0] for w in words)
    return len(words) > 1 and re.search(rf"\b{initials}\b", passage) is not None


def _evidence_match(evidence: str, passage: str) -> str | None:
    """How the evidence quote matches the (squashed) passage, or None if it does not.

    "exact": the quote occurs as is. "pieces": the quote joins parts with "..." and every
    part occurs. "words": the quote is reworded, but at least WORD_SHARE of its words occur
    in the passage. Some models paraphrase their quotes; the fact can still be right.
    """
    q = _squash(evidence)
    if q in passage:
        return "exact"
    pieces = [x.strip(" .,;:") for x in re.split(r"\.\.\.|…", q)]
    pieces = [x for x in pieces if x]
    if len(pieces) > 1 and all(x in passage for x in pieces):
        return "pieces"
    words = re.findall(r"[a-z0-9]+", q)
    have = set(re.findall(r"[a-z0-9]+", passage))
    if words and sum(w in have for w in words) / len(words) >= WORD_SHARE:
        return "words"
    return None


def _parse_node(token: str, filer: str) -> tuple[str, str, str | None] | None:
    """Turn "SELF", "Type:Name" or "Risk:category:name" into (type, name, category), or None."""
    if token == "SELF":
        return "Company", filer, None
    typ, _, rest = token.partition(":")
    if not rest:
        return None
    if typ == "Risk":
        cat, _, name = rest.partition(":")
        return ("Risk", name.strip(), cat.strip()) if name.strip() else None
    return typ, rest.strip(), None


def validate_b(facts: list, chunk_text: str, filer: str, stats: dict) -> list[dict]:
    """Keep only facts that fit the schema, and mark each kept fact as verified or not.

    `chunk_text` is the chunk's heading and text, as the LLM saw them.
    A fact is verified when its evidence quote matches the chunk (see `_evidence_match`) and
    every named entity in it other than the filer (company, product, segment, place) occurs
    in the chunk. The name check is the main guard against made-up facts: a model that adds a
    competitor from its own knowledge fails it. Unverified facts are kept but flagged.

    A fact repeated within the chunk (same source, relation and target, ignoring case) is
    kept once, preferring a copy whose evidence quote was found; extra copies are counted as
    duplicates. The graph needs each fact once per chunk; repeats would only inflate counts.
    """
    out: dict[tuple, dict] = {}
    passage = _squash(chunk_text)
    for f in facts:
        if not (isinstance(f, list) and len(f) == 4 and all(isinstance(x, str) for x in f)):
            stats["malformed"] += 1
            continue
        s, rel, o, evidence = f
        src, dst = _parse_node(s, filer), _parse_node(o, filer)
        if not src or not dst:
            stats["malformed"] += 1
            continue
        if src[0] not in ENTITY_TYPES or dst[0] not in ENTITY_TYPES:
            stats["bad_entity_type"] += 1
            continue
        if (src[0], rel, dst[0]) not in VALIDATION_SCHEMA:
            stats["not_in_schema"] += 1
            continue
        if any(n[0] == "Risk" and n[2] not in RISK_CATEGORIES for n in (src, dst)):
            stats["bad_category"] += 1
            continue
        match = _evidence_match(evidence, passage)
        # Every named entity other than the filer must be named in the passage itself.
        names_found = all(_name_in(n[1], passage) for n, tok in ((src, s), (dst, o))
                          if tok != "SELF" and n[0] in _NAMED_TYPES)
        key = (src[0], src[1].lower(), rel, dst[0], dst[1].lower())
        if key in out:
            stats["duplicates"] += 1
            if match and not out[key]["evidence_match"]:
                out[key].update(evidence=evidence, evidence_match=match)
            continue
        out[key] = {"source": src[1], "source_type": src[0], "relation": rel, "target": dst[1],
                    "target_type": dst[0], **({"category": dst[2] or src[2]} if "Risk" in (src[0], dst[0]) else {}),
                    "evidence": evidence, "evidence_match": match, "names_found": names_found}
    for f in out.values():
        f["verified"] = bool(f["evidence_match"]) and f["names_found"]
        stats["facts"] += 1
        stats[f"evidence_{f['evidence_match'] or 'not_found'}"] += 1
        stats["names_not_found"] += not f["names_found"]
        stats["unverified"] += not f["verified"]
        stats["evidence_over_10_words"] += len(f["evidence"].split()) > 10
    return list(out.values())


# --------------------------------------------------------------------------------------
# Running the prototype on one filing
# --------------------------------------------------------------------------------------
def run(ticker: str, fy: int, backend: str = "claude_cli",
        max_a_calls: int | None = None, max_b_calls: int | None = None) -> None:
    """Extract routes A and B for one filing, validate, save and print a summary."""
    from collections import Counter

    llm = get_llm(backend)
    titles_per_call, chunks_per_call, max_out = BATCH_BY_BACKEND.get(backend, BATCH_DEFAULT)
    name = _filer_name(ticker)
    row, titles, chunks = filing_inputs(ticker, fy)
    print(f"{ticker} FY{fy} ({row.accession_number}) via {backend}/{llm.model}: "
          f"{len(titles)} new risk titles, {len(chunks)} route-B chunks")
    EXTRACT_PROTO_DIR.mkdir(parents=True, exist_ok=True)
    usage = Counter()

    def account(res: CallResult, route: str) -> None:
        usage[f"{route}_calls"] += 1
        usage[f"{route}_cached"] += res.cached
        usage[f"{route}_in"] += res.response.input_tokens
        usage[f"{route}_out"] += res.response.output_tokens
        usage[f"{route}_secs"] += res.seconds
        usage[f"{route}_bad_json"] += res.data is None

    # Route A
    out_a, stats_a = [], Counter()
    batches = [titles[i:i + titles_per_call] for i in range(0, len(titles), titles_per_call)]
    for batch in batches[:max_a_calls]:
        res = call_cached(llm, SYSTEM_A, prompt_a(name, ticker, fy, batch), max_tokens=max_out)
        account(res, "A")
        got = {it.get("id"): it for it in (res.data or {}).get("risks", [])}
        for i, title in enumerate(batch, 1):
            it = got.get(i)
            if not it:
                stats_a["missing"] += 1
                continue
            if it.get("category") not in RISK_CATEGORIES:
                stats_a["bad_category"] += 1
                continue
            out_a.append({"title": title, "risk": it.get("risk"), "category": it["category"],
                          "places": it.get("places", [])})
    # Route B
    out_b, stats_b = [], Counter()
    groups = [chunks[i:i + chunks_per_call] for i in range(0, len(chunks), chunks_per_call)]
    for group in groups[:max_b_calls]:
        res = call_cached(llm, SYSTEM_B, prompt_b(name, ticker, fy, group), max_tokens=max_out)
        account(res, "B")
        got = res.data if isinstance(res.data, dict) else {}
        for i, c in enumerate(group, 1):
            facts = got.get(str(i))
            if facts is None:
                stats_b["missing_passage"] += 1
                continue
            out_b.append({"chunk_id": c.chunk_id, "section": c.section, "heading": c.heading,
                          "facts": validate_b(facts if isinstance(facts, list) else [],
                                                  f"{c.heading or ''}\n{c.text}", name, stats_b)})

    meta = {"ticker": ticker, "fiscal_year": fy, "accession": row.accession_number,
            "backend": backend, "model": llm.model, "prompt_version": PROMPT_VERSION,
            "schema_version": SCHEMA_VERSION}
    for route, items, stats in (("A", out_a, stats_a), ("B", out_b, stats_b)):
        (EXTRACT_PROTO_DIR / f"{row.accession_number}_{route}.json").write_text(json.dumps(
            {**meta, "stats": dict(stats), "items": items}, indent=1))

    for route in ("A", "B"):
        if usage[f"{route}_calls"]:
            k = usage[f"{route}_calls"]
            print(f"Route {route}: {k} calls ({usage[f'{route}_cached']} cached, "
                  f"{usage[f'{route}_bad_json']} bad JSON), input {usage[f'{route}_in']:,} / output "
                  f"{usage[f'{route}_out']:,} tokens ({usage[f'{route}_in'] // k:,} / "
                  f"{usage[f'{route}_out'] // k:,} per call), {usage[f'{route}_secs'] / k:.0f} s per call")
    print(f"Route A: {len(out_a)} risks kept; problems {dict(stats_a)}")
    print(f"Route B: {len(out_b)} passages; {dict(stats_b)}")
    print(f"Saved to {EXTRACT_PROTO_DIR}/{row.accession_number}_{{A,B}}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("fiscal_year", type=int)
    ap.add_argument("--backend", default="claude_cli", help="groq, antigravity_cli or claude_cli")
    ap.add_argument("--max-a-calls", type=int, help="stop route A after this many calls")
    ap.add_argument("--max-b-calls", type=int, help="stop route B after this many calls")
    a = ap.parse_args()
    run(a.ticker, a.fiscal_year, a.backend, a.max_a_calls, a.max_b_calls)
