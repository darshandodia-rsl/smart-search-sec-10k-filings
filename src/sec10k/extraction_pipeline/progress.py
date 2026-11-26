"""Stage 7 of the extraction pipeline: progress reports and events for phone pushes.

What it does:
  - `emit()` appends notable run events to an events file (one JSON object per line), e.g. a
    backend's gate closed or reopened, Sonnet needs an allowance, a company is finished,
    errors in a row, a backend has no work left, the run stopped, and an hourly summary.
    Events worth a phone push carry `"push": true`. The Claude session watching a run reads
    this file and sends those pushes (the Python runner cannot send them itself).
  - `status_report()` builds a short text report from the job store: overall % done, a rough
    ETA, and per backend its companies, items done, calls, tokens and errors in the last 24
    hours, plus its latest gate state from the events file.

Why it is needed: the full run takes weeks and mostly runs unattended. The user needs to see
progress and be told when something needs them (a Sonnet allowance, repeated errors)
without reading logs; one push per LLM call would be far too many, so only events push.

The ETA uses the items finished in the last few hours, so it is rough: quotas make the pace
uneven (e.g. fast while Groq's daily budget lasts, slow after).

Run: .venv/bin/python -m sec10k.extraction_pipeline.progress   (prints the status report)
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sec10k.config import EXTRACTION_EVENTS, JOBS_DB
from sec10k.extraction_pipeline.jobs import MAX_ATTEMPTS, JobStore

RATE_WINDOW_HOURS = 3          # the ETA uses the pace of the last this many hours
SUMMARY_EVERY_MIN = 60         # the runner emits a summary event this often
ALLOWANCE_REMINDER_MIN = 60    # a waiting Sonnet worker repeats its allowance request this often
ERRORS_IN_A_ROW = 3            # failed calls in a row on one backend that trigger an event

_lock = threading.Lock()


def emit(kind: str, message: str, push: bool, backend: str | None = None,
         path: Path | None = None, **data) -> None:
    """Append one event to the events file. `push` marks events the watcher should send to the phone.

    Thread-safe: the worker threads of all backends write to the same file. `path` defaults
    to EXTRACTION_EVENTS (looked up at call time, so tests can point it elsewhere).
    """
    path = path or EXTRACTION_EVENTS
    event = {"at": datetime.now().astimezone().isoformat(timespec="seconds"), "kind": kind,
             "backend": backend, "push": push, "message": message, **data}
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(event) + "\n")


def last_events(path: Path | None = None, n: int = 500) -> list[dict]:
    """The last `n` events (oldest first), for status reports. Empty if there is no events file yet."""
    path = path or EXTRACTION_EVENTS
    if not path.exists():
        return []
    lines = path.read_text().splitlines()[-n:]
    return [json.loads(x) for x in lines if x.strip()]


def _since(hours: float) -> str:
    """UTC ISO timestamp `hours` ago, comparable with the job store's timestamps."""
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(timespec="seconds")


def status_report(store: JobStore, events_path: Path | None = None) -> str:
    """A short text report of the whole run: % done, ETA, and one line per backend."""
    q = lambda sql, *p: store.db.execute(sql, p).fetchall()
    st = dict(q("SELECT status, COUNT(*) FROM items WHERE kind = 'extract' GROUP BY status"))
    total = sum(st.values())
    done = st.get("done", 0)
    stuck = q("SELECT COUNT(*) FROM items WHERE status = 'failed' AND attempts >= ?", MAX_ATTEMPTS)[0][0]
    left = total - done - stuck
    recent = q("SELECT COUNT(*) FROM results WHERE created_at >= ?", _since(RATE_WINDOW_HOURS))[0][0]
    rate = recent / RATE_WINDOW_HOURS
    eta = f"~{left / rate:.0f} h at the last {RATE_WINDOW_HOURS} h's pace" if rate and left else "n/a"
    companies = q("SELECT COUNT(*), SUM(backend IS NOT NULL) FROM companies")[0]
    lines = [f"Extraction: {done:,} / {total:,} items done ({done / total:.0%})" if total else "Extraction: no items",
             f"  left {left:,} (pending {st.get('pending', 0):,}, running {st.get('running', 0):,}, "
             f"retrying {st.get('failed', 0) - stuck:,}), failed for good {stuck:,}",
             f"  pace {rate:.0f} items/h, ETA {eta}; companies claimed {companies[1] or 0} / {companies[0]}"]
    gates = {}
    for e in last_events(events_path):
        if e.get("backend") and e["kind"] in ("gate_closed", "gate_opened", "needs_allowance", "backend_idle", "backend_stopped"):
            gates[e["backend"]] = f"{e['kind']} {e['at'][11:16]}: {e['message']}"
    day = _since(24)
    for backend, calls, cached, toks, errs in q(
            "SELECT backend, COUNT(*), SUM(cached), SUM(input_tokens + output_tokens), SUM(error IS NOT NULL)"
            " FROM calls WHERE at >= ? GROUP BY backend ORDER BY backend", day):
        owned = q("SELECT GROUP_CONCAT(ticker, ' ') FROM companies WHERE backend = ?", backend)[0][0]
        items = q("SELECT COUNT(*) FROM results WHERE backend = ? AND created_at >= ?", backend, day)[0][0]
        lines.append(f"  {backend}: last 24 h {calls} calls ({cached} cached, {errs} errors), {toks:,} tokens, "
                     f"{items:,} items; companies: {owned or '-'}")
        if backend in gates:
            lines.append(f"      {gates[backend]}")
    for backend in sorted(set(gates) - {r[0] for r in q("SELECT DISTINCT backend FROM calls WHERE at >= ?", day)}):
        lines.append(f"  {backend}: no calls in 24 h; {gates[backend]}")
    return "\n".join(lines)


if __name__ == "__main__":
    if not JOBS_DB.exists():
        raise SystemExit(f"{JOBS_DB} does not exist: no extraction run yet")
    print(status_report(JobStore()))
