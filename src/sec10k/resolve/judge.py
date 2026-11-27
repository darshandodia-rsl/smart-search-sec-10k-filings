"""Stage 5 of entity resolution: the LLM judge for unclear pairs of names.

What it does:
  - `SYSTEM` is the judge's instructions: for each pair of names (A, B) pick one decision,
    with the direction spelled out (A_PRODUCT_OF_B or B_PRODUCT_OF_A, ...), so the model never
    has to guess which side is the owner.
  - `describe()` gives the model context for one name: its type, other spellings, how the
    10-Ks used it (relations), which filers named it and a few evidence quotes.
  - `batches()` reads the pairs marked `to_llm` in `data/resolution/candidate_pairs.csv` that
    have no decision yet and groups them into calls, pairs that share a name next to each other.
  - `run()` sends the calls through the LLM cache, checks every answer, and saves one decision
    per pair in `data/resolution/decisions.csv` after each call. Product and subsidiary answers
    are stored owned-first: (a, b, PRODUCT_OF) means a is b's product, as in `overrides.csv`.
    Before each call it asks the extraction router's gate for the backend (budgets, Sonnet
    schedule, Antigravity limits) and stops when the gate is closed or the budget runs out.
  - `--dry-run` prints one real batch and the estimated tokens of all calls. It calls no LLM.

Why it is needed: rules and embeddings can say two names look alike, but not whether
"Salesforce.org" is Salesforce (it is not: it is the nonprofit arm) or whether
"Amazon Web Services" is a product or a subsidiary of Amazon. That needs world knowledge.
The judge sees only the few hundred pairs that earlier stages could not settle, and when
unsure it answers DIFFERENT: a missed merge leaves two nodes, a wrong merge mixes up the
facts of two real things, which is worse.

Decisions are kept per pair, not per call: when new filings bring new pairs, the batches
change, so a per-call cache would miss; with per-pair decisions a re-run only sends pairs that
were never decided. The LLM cache stays as a second safety net for a call that is repeated.
Only start a run when the user says so (it spends quota).

Run: .venv/bin/python -m sec10k.resolve.judge --dry-run [--batch N]
     .venv/bin/python -m sec10k.resolve.judge --run --backend groq|antigravity_cli|claude_cli [--max-calls N]
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from datetime import datetime

import pandas as pd

from sec10k.config import RESOLUTION_DIR
from sec10k.extract.llm_extract import call_cached, drop_cached, filer_name
from sec10k.llm.providers import BudgetExceeded, LLMProvider, get_llm
from sec10k.resolve.names import Key, load_overrides, match_aliases, name_inventory

JUDGE_PROMPT_VERSION = 1   # raise when SYSTEM or the pair format changes; stored with each decision
DECISIONS_FILE = RESOLUTION_DIR / "decisions.csv"
DECISION_COLUMNS = ["a", "b", "decision", "reason", "raw", "a_name", "b_name", "backend", "model",
                    "prompt_version", "decided_at"]
BATCH_SIZE = 25            # pairs per call
# (pairs per call, max output tokens). Groq's free tier counts input + allowed output against
# 8K tokens per minute, so a Groq call must stay well under 8K.
BATCH_BY_BACKEND = {"groq": (20, 3000)}
BATCH_DEFAULT = (BATCH_SIZE, 6000)
RETRY_WAITS = [10, 30, 60]   # seconds between retries of a failed call
JUDGE_DECISIONS = ("SAME", "A_PRODUCT_OF_B", "B_PRODUCT_OF_A", "A_SUBSIDIARY_OF_B",
                   "B_SUBSIDIARY_OF_A", "DIFFERENT")

SYSTEM = """You decide whether two names, extracted from US tech companies' annual reports (10-K),
refer to the same real-world entity. For each numbered pair (A, B) choose exactly one decision:

- SAME: A and B are the same company, or the same product: another spelling, an abbreviation,
  a former name, a legal-name variant, a version number or a typo.
  "Cisco" / "Cisco Systems, Inc."; "Prisma Access 3.0" / "Prisma Access"; "AutoCAD Civil3D" / "AutoCAD Civil 3D"
- A_PRODUCT_OF_B / B_PRODUCT_OF_A: one is a product, service or brand of the other, which is a
  company. "Microsoft Azure" / "Microsoft" -> A_PRODUCT_OF_B
- A_SUBSIDIARY_OF_B / B_SUBSIDIARY_OF_A: one is a company owned by the other (a subsidiary, or an
  acquired company that still operates under its own name).
  "Amazon Web Services" / "Amazon" -> A_SUBSIDIARY_OF_B; "LinkedIn" / "Microsoft" -> A_SUBSIDIARY_OF_B
- DIFFERENT: anything else: different companies, different products, different editions or tiers
  of a product ("TurboTax Live" / "TurboTax Online"), a part of a product ("Azure Synapse" /
  "Azure"), a generic description ("cloud platform"), or a nonprofit or foundation named after
  a company ("Salesforce.org" / "Salesforce.com").

Rules:
1. Use your knowledge of companies and products. The context lines show how the reports used
   each name; the evidence quotes come from the reports.
2. If you are not sure, answer DIFFERENT. Wrongly merging two things is worse than missing a merge.
3. The type shown (Company / Product) comes from an automatic extraction and may be wrong;
   decide by what the thing really is.

Answer with JSON only, no other text, one item per pair, "reason" at most 10 words:
{"pairs": [{"id": 1, "decision": "SAME", "reason": "legal-name variant"}, ...]}"""


def describe(k: Key) -> str:
    """Context lines for one name: type, other spellings, corpus company (if matched), uses, filers, evidence."""
    other = [v for v, _ in k.variants.most_common() if v != k.name][:3]
    corpus = f"; this is the listed company {filer_name(k.ticker)} ({k.ticker})" if k.ticker else ""
    lines = [f'"{k.name}" ({k.label}{"; also written: " + ", ".join(other) if other else ""}{corpus})']
    uses = ", ".join(f"{u} x{n}" for u, n in k.uses.most_common(3))
    names = sorted(filer_name(t) for t in k.filers)
    named_by = ", ".join(names[:4]) + (f" +{len(names) - 4} more" if len(names) > 4 else "")
    lines.append(f"     used as: {uses}; named by: {named_by}")
    if k.evidence:
        lines.append("     evidence: " + " | ".join(f'"{e}"' for e in k.evidence[:2]))
    return "\n".join(lines)


def prompt(pairs: list[tuple[str, str]], keys: dict[str, Key]) -> str:
    """The user prompt of one call: the numbered pairs, each name with its context."""
    parts = []
    for i, (a, b) in enumerate(pairs, 1):
        parts.append(f"Pair {i}\n  A: {describe(keys[a])}\n  B: {describe(keys[b])}")
    return "\n\n".join(parts)


def load_decisions() -> pd.DataFrame:
    """The saved per-pair decisions (empty table if none yet)."""
    if not DECISIONS_FILE.exists():
        return pd.DataFrame(columns=DECISION_COLUMNS)
    return pd.read_csv(DECISIONS_FILE, dtype=str, keep_default_na=False)


def save_decisions(df: pd.DataFrame) -> None:
    """Write the decisions file in one go (temporary file, then rename), so a crash never leaves half a file."""
    RESOLUTION_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DECISIONS_FILE.with_suffix(".tmp")
    df[DECISION_COLUMNS].to_csv(tmp, index=False)
    tmp.replace(DECISIONS_FILE)


def batches(keys: dict[str, Key], size: int = BATCH_SIZE) -> list[list[tuple[str, str]]]:
    """The `to_llm` pairs still without a decision (saved or hand override), in calls of `size`.

    Pairs are sorted by their B name, then A, so pairs about one company sit in the same call
    (the model answers them consistently) and the same input always gives the same batches.
    """
    df = pd.read_csv(RESOLUTION_DIR / "candidate_pairs.csv", keep_default_na=False)
    df = df[(df.to_llm.astype(str) == "True") & df.a.isin(keys) & df.b.isin(keys)]
    d = load_decisions()
    done = {frozenset(p) for p in zip(d.a, d.b)} | {frozenset(p) for p in load_overrides()}
    pairs = sorted((p for p in zip(df.a, df.b) if frozenset(p) not in done), key=lambda p: (p[1], p[0]))
    return [pairs[i:i + size] for i in range(0, len(pairs), size)]


def parse_answer(data: dict | None, pairs: list[tuple[str, str]], keys: dict[str, Key]) -> tuple[list[dict], int]:
    """Turn one call's JSON into decision rows; returns (rows, number of pairs without a valid answer).

    An item with an unknown id or decision is skipped, and a pair answered twice keeps its first
    answer; skipped pairs get no row and are asked again on the next run.
    """
    items = data.get("pairs") if isinstance(data, dict) else None
    rows, seen = [], set()
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        i, raw = it.get("id"), it.get("decision")
        if not isinstance(i, int) or not 1 <= i <= len(pairs) or i in seen or raw not in JUDGE_DECISIONS:
            continue
        seen.add(i)
        a, b = pairs[i - 1]
        if raw.startswith("B_"):        # store owned-first: (a, b, PRODUCT_OF) = a is b's product
            a, b = b, a
        decision = raw[2:-2] if raw[:2] in ("A_", "B_") else raw
        rows.append({"a": a, "b": b, "decision": decision, "reason": str(it.get("reason", ""))[:200],
                     "raw": raw, "a_name": keys[a].name, "b_name": keys[b].name})
    return rows, len(pairs) - len(rows)


def call_with_retries(llm: LLMProvider, user: str, max_tokens: int):
    """One cached call; on an error (not a budget stop) wait and try again, up to len(RETRY_WAITS) times."""
    for attempt, wait in enumerate([0] + RETRY_WAITS):
        time.sleep(wait)
        try:
            return call_cached(llm, SYSTEM, user, max_tokens)
        except BudgetExceeded:
            raise
        except Exception as e:
            print(f"    call error ({type(e).__name__}: {str(e)[:120]}), attempt {attempt + 1}")
    raise RuntimeError("call failed after all retries")


def run(backend: str, max_calls: int | None = None) -> None:
    """Judge every undecided `to_llm` pair on `backend`, saving decisions after each call.

    Stops (pairs stay undecided, a re-run continues) when the router's gate for the backend is
    closed, the daily budget is used up, or a call keeps failing.
    """
    from sec10k.extraction_pipeline.router import Router

    llm = get_llm(backend)
    size, max_tokens = BATCH_BY_BACKEND.get(backend, BATCH_DEFAULT)
    keys = name_inventory()
    match_aliases(keys)
    calls = batches(keys, size)[:max_calls]
    print(f"{backend} ({llm.model}): {sum(len(c) for c in calls)} pairs to judge in {len(calls)} calls of up to {size}")
    router, decided = Router(None), load_decisions()
    totals = Counter()
    for n, pairs in enumerate(calls, 1):
        gate = router.gate(llm)
        if not gate.allowed:
            print(f"Stopped: {backend} gate closed ({gate.reason}). Re-run later to continue.")
            break
        user = prompt(pairs, keys)
        try:
            res = call_with_retries(llm, user, max_tokens)
        except BudgetExceeded as e:
            print(f"Stopped: budget ({e}). Re-run later to continue.")
            break
        except RuntimeError as e:
            print(f"Stopped: {e}. Re-run later to continue.")
            break
        rows, missing = parse_answer(res.data, pairs, keys)
        if not rows:
            drop_cached(llm, SYSTEM, user)   # unusable answer: never serve it from the cache again
        stamp = {"backend": backend, "model": llm.model, "prompt_version": str(JUDGE_PROMPT_VERSION),
                 "decided_at": datetime.now().isoformat(timespec="seconds")}
        if rows:
            decided = pd.concat([decided, pd.DataFrame([r | stamp for r in rows])], ignore_index=True)
            save_decisions(decided)
        totals.update(r["decision"] for r in rows)
        totals["missing"] += missing
        totals["tokens"] += 0 if res.cached else res.response.total_tokens
        print(f"  call {n}/{len(calls)}: {len(rows)}/{len(pairs)} decided, "
              f"{res.response.total_tokens:,} tokens{' (cached)' if res.cached else ''}, {res.seconds:.0f} s")
    print(f"\nDecisions this run: { {k: v for k, v in totals.items() if k not in ('missing', 'tokens')} }, "
          f"{totals['missing']} pairs without a valid answer, {totals['tokens']:,} new tokens. "
          f"All saved decisions: {len(decided)} in {DECISIONS_FILE}")


def dry_run(size: int) -> None:
    """Print the system prompt, one real batch and the estimated tokens of all calls (no LLM call)."""
    keys = name_inventory()
    match_aliases(keys)
    calls = batches(keys, size)
    user = [prompt(c, keys) for c in calls]
    est = [(len(SYSTEM) + len(u)) // 4 for u in user]   # ~4 characters per token, as the budgets do
    pick = next((i for i, c in enumerate(calls) if any("amazon" in p[1] for p in c)), 0)
    print("=== SYSTEM ===\n" + SYSTEM)
    print(f"\n=== USER (call {pick + 1} of {len(calls)}, first 6 of {len(calls[pick])} pairs) ===")
    print(prompt(calls[pick][:6], keys))
    print(f"\n{sum(len(c) for c in calls)} pairs in {len(calls)} calls of up to {size}; "
          f"input ≈ {sum(est):,} tokens (≈ {min(est):,}–{max(est):,} per call, estimate: chars / 4); "
          f"output ≈ 25 tokens per pair + the model's reasoning")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="print a batch, call no LLM")
    mode.add_argument("--run", action="store_true", help="judge the undecided pairs (spends quota)")
    ap.add_argument("--backend", help="LLM backend for --run: groq, antigravity_cli or claude_cli")
    ap.add_argument("--batch", type=int, default=BATCH_SIZE, help="pairs per call for --dry-run")
    ap.add_argument("--max-calls", type=int, help="stop after this many calls (--run)")
    args = ap.parse_args()
    if args.run:
        if not args.backend:
            ap.error("--run needs --backend")
        run(args.backend, args.max_calls)
    else:
        dry_run(args.batch)
