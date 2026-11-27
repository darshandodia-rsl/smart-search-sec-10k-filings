"""Re-check stored route-B facts with the current name check, without asking the LLM again.

What it does: for every route-B result in the job store, rebuilds the chunk text the LLM
saw, runs the current name check (`llm_extract._name_in`) on each fact's named entities
and recomputes the fact's `names_found` and `verified` flags. The evidence check result
(`evidence_match`) is kept as stored. A dry run (the default) prints what would change;
`--apply` saves the changed payloads in one transaction.

Why it is needed: `verified` is computed once, when an answer arrives, and stored with the
result. When the check itself improves, results already in the store keep their old flags.
This brings them in line at no LLM cost; afterwards `python -m sec10k.graph.facts` copies
the new flags into the graph. The filer counts as named (the LLM writes it as "SELF"), as
in `validate_b`.

`--apply` refuses while an extraction runner is alive or any item is "running", because
the runner writes to the same results table.

Run: .venv/bin/python -m sec10k.extract.revalidate [TICKER ...] [--apply]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from collections import Counter

from sec10k.config import JOBS_DB
from sec10k.extract.llm_extract import _NAMED_TYPES, _name_in, _squash, filer_name
from sec10k.ingest.chunker import iter_chunks


def recheck(facts: list[dict], chunk_text: str, filer: str) -> int:
    """Recompute `names_found` and `verified` of one chunk's facts in place; return how many changed."""
    passage = _squash(chunk_text)
    changed = 0
    for f in facts:
        names = [(f["source"], f["source_type"]), (f["target"], f["target_type"])]
        found = all(_name_in(n, passage, chunk_text) for n, typ in names
                    if typ in _NAMED_TYPES and not (typ == "Company" and n == filer))
        verified = bool(f["evidence_match"]) and found
        if (found, verified) != (f["names_found"], f["verified"]):
            f["names_found"], f["verified"] = found, verified
            changed += 1
    return changed


def runner_alive() -> bool:
    """True if an extraction runner process is running (it writes to the job store)."""
    return subprocess.run(["pgrep", "-f", "extraction_pipeline.worker"], capture_output=True).returncode == 0


def run(tickers: list[str] | None, apply: bool) -> None:
    """Re-check the route-B results of the given companies (default: all) and print or save the changes."""
    db = sqlite3.connect(JOBS_DB, timeout=30)
    if apply and (runner_alive() or db.execute("SELECT 1 FROM items WHERE status = 'running'").fetchone()):
        raise SystemExit("an extraction runner is active: stop it first (/wind-down), then re-run with --apply")
    if not tickers:
        tickers = [r[0] for r in db.execute(
            "SELECT DISTINCT i.ticker FROM items i JOIN results r USING (item_id) WHERE i.route = 'B' ORDER BY 1")]
    updates, totals, examples = [], Counter(), []
    for t in tickers:
        rows = db.execute("SELECT i.item_id, i.chunk_id, r.payload FROM items i JOIN results r USING (item_id)"
                          " WHERE i.ticker = ? AND i.route = 'B'", (t,)).fetchall()
        want = {cid for _, cid, _ in rows}
        texts = {c.chunk_id: f"{c.heading or ''}\n{c.text}" for c in iter_chunks([t]) if c.chunk_id in want}
        filer, n_changed = filer_name(t), 0
        for item_id, chunk_id, payload in rows:
            if chunk_id not in texts:
                totals["chunk_missing"] += 1
                continue
            p = json.loads(payload)
            totals["facts"] += len(p["facts"])
            before = [f["verified"] for f in p["facts"]]
            n = recheck(p["facts"], texts[chunk_id], filer)
            if n:
                updates.append((json.dumps(p), item_id))
                n_changed += n
                for f, was in zip(p["facts"], before):
                    totals["verified_to_unverified"] += was and not f["verified"]
                    totals["unverified_to_verified"] += f["verified"] and not was
                    if was != f["verified"] and len(examples) < 20:
                        examples.append(f"{t}: {f['source']} -{f['relation']}-> {f['target']}  \"{f['evidence']}\"")
        totals["facts_changed"] += n_changed
        if n_changed:
            print(f"{t}: {n_changed} facts changed")
    print(f"\n{dict(totals)}")
    for e in examples:
        print("  " + e)
    if apply and updates:
        with db:
            db.executemany("UPDATE results SET payload = ? WHERE item_id = ?", updates)
        print(f"\nSaved {len(updates)} results. Next: .venv/bin/python -m sec10k.graph.facts")
    elif updates:
        print(f"\nDry run: {len(updates)} results would change. Re-run with --apply to save.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Re-check stored route-B facts with the current name check (no LLM).")
    ap.add_argument("tickers", nargs="*", help="only these companies (default: all with route-B results)")
    ap.add_argument("--apply", action="store_true", help="save the changes (default: dry run)")
    a = ap.parse_args()
    run(a.tickers or None, a.apply)
