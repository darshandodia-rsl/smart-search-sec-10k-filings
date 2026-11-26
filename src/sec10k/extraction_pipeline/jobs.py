"""Stage 2 of the extraction pipeline: the job store, which remembers the status of every work item.

What it does: keeps one row per planner `WorkItem` in a SQLite database file (a small
database in one file, built into Python). Each row has a status:
  - pending: waiting for an LLM call
  - running: inside an LLM call right now
  - done:    extracted; its facts are saved
  - failed:  the call or its validation failed; retried later, up to MAX_ATTEMPTS
  - reuse:   needs no LLM call; its facts are copied from its `source` item
Every status change is committed at once, so the file always matches what really happened.
Next to the items, the same file holds:
  - companies: each company's priority and the backend that owns it (the router's taxi rank)
  - results:   the extracted payload of every done item, saved in the same transaction that
               marks it done (the graph writer reads these)
  - calls:     one row per LLM call (tokens, time, outcome), for progress reports and ETAs

Why it is needed: the full run takes weeks and will be interrupted (crashes, sleep, quota
limits). Because the status lives on disk, a restarted runner continues where it stopped:
`recover()` puts items a crash left "running" back to "pending", and finished items are never
sent again. Adding a wave is idempotent: items are keyed by the planner's stable `item_id`,
so adding the same wave twice changes nothing and never resets finished work.

The database uses WAL mode (write-ahead log), so the status command can read it safely while
the runner is writing.

Run:
  .venv/bin/python -m sec10k.extraction_pipeline.jobs add MSFT CRM [--first-years N]
  .venv/bin/python -m sec10k.extraction_pipeline.jobs replan MSFT CRM [--first-years N] [--requeue [--route A|B]]
  .venv/bin/python -m sec10k.extraction_pipeline.jobs status
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from sec10k.config import JOBS_DB
from sec10k.extraction_pipeline.planner import WorkItem, plan_wave

MAX_ATTEMPTS = 3   # a failed item is retried until it has been tried this many times
# SQL condition for "ready for an LLM call": pending, or failed with attempts left (takes MAX_ATTEMPTS).
_READY = "(status = 'pending' OR (status = 'failed' AND attempts < ?))"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    item_id     TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,              -- extract | reuse
    route       TEXT NOT NULL,              -- A (risk title) | B (chunk)
    ticker      TEXT NOT NULL,
    fiscal_year INTEGER NOT NULL,
    accession   TEXT NOT NULL,
    chunk_id    TEXT NOT NULL,
    title       TEXT,                       -- route A only
    source      TEXT,                       -- reuse only: item_id of the first copy
    tokens      INTEGER NOT NULL,
    status      TEXT NOT NULL,              -- pending | running | done | failed | reuse
    attempts    INTEGER NOT NULL DEFAULT 0,
    last_error  TEXT,
    backend     TEXT,                       -- backend that finished (or last tried) the item
    added_at    TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS items_queue ON items (status, route, ticker, fiscal_year);
CREATE TABLE IF NOT EXISTS companies (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,  -- priority: order in which companies were added
    ticker      TEXT UNIQUE NOT NULL,
    backend     TEXT,                               -- backend that owns the company (NULL = not yet claimed)
    assigned_at TEXT
);
CREATE TABLE IF NOT EXISTS results (
    item_id        TEXT PRIMARY KEY REFERENCES items (item_id),
    payload        TEXT NOT NULL,   -- JSON: route A {title, risk, category, places}; route B {facts: [...]}
    backend        TEXT NOT NULL,
    model          TEXT NOT NULL,
    prompt_version INTEGER NOT NULL,
    created_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    backend       TEXT NOT NULL,
    route         TEXT NOT NULL,
    ticker        TEXT NOT NULL,
    n_items       INTEGER NOT NULL,
    n_done        INTEGER NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    seconds       REAL NOT NULL,
    cached        INTEGER NOT NULL,  -- 1 if the answer came from the LLM cache (no quota spent)
    error         TEXT,              -- set when the call itself failed
    at            TEXT NOT NULL
);
"""


def _now() -> str:
    """Current UTC time as an ISO string, stored on every status change for progress reports."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobStore:
    """The job store: one SQLite file holding the status of every extraction work item.

    The batcher reads pending items from it, the worker marks them done or failed, and the
    status command and progress reports count them.
    """

    def __init__(self, path: Path = JOBS_DB):
        """Open (and create if needed) the job store at `path`."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(_SCHEMA)

    def add(self, items: list[WorkItem]) -> int:
        """Add planned items; items already in the store are left untouched. Returns how many were new.

        Also registers each company, in the order it first appears, which sets its priority:
        companies of earlier waves (and earlier in a wave's ticker list) are claimed first.
        """
        now = _now()
        with self.db:
            self.db.executemany("INSERT OR IGNORE INTO companies (ticker) VALUES (?)",
                                [(t,) for t in dict.fromkeys(i.ticker for i in items)])
            before = self.db.total_changes   # count only item rows below
            self.db.executemany(
                "INSERT OR IGNORE INTO items (item_id, kind, route, ticker, fiscal_year, accession, chunk_id,"
                " title, source, tokens, status, added_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(i.item_id, i.kind, i.route, i.ticker, i.fiscal_year, i.accession, i.chunk_id, i.title,
                  i.source, i.tokens, "pending" if i.kind == "extract" else "reuse", now, now)
                 for i in items])
        return self.db.total_changes - before

    def replan(self, items: list[WorkItem], requeue: bool, route: str | None = None) -> dict[str, int]:
        """Bring the store in line with a new plan of some filings, e.g. after a prompt or schema change.

        For the filings in `items` only (other filings are never touched): items the new plan
        no longer contains are deleted together with their results (e.g. titles a planning
        rule now skips), new ones are added, and with `requeue` every done or failed extract
        item goes back to pending with its attempts reset, so the next run extracts it again
        with the current prompt (its old result is replaced when the new one is saved).
        `route` ("A" or "B") limits the requeue to that route, e.g. when only the route-B
        prompt changed; the other route's results are kept.
        Refuses while any of those filings has items running, since a runner is working on them.
        Returns counts of removed, added and requeued items.
        """
        accessions = sorted({i.accession for i in items})
        marks = ",".join("?" * len(accessions))
        if self.db.execute(f"SELECT COUNT(*) FROM items WHERE accession IN ({marks}) AND status = 'running'",
                           accessions).fetchone()[0]:
            raise RuntimeError("some of these filings have items running: stop the runner first")
        planned = {i.item_id for i in items}
        stale = [r[0] for r in self.db.execute(f"SELECT item_id FROM items WHERE accession IN ({marks})", accessions)
                 if r[0] not in planned]
        with self.db:
            self.db.executemany("DELETE FROM results WHERE item_id = ?", [(i,) for i in stale])
            self.db.executemany("DELETE FROM items WHERE item_id = ?", [(i,) for i in stale])
        added = self.add(items)
        requeued = 0
        if requeue:
            with self.db:
                requeued = self.db.execute(
                    "UPDATE items SET status = 'pending', attempts = 0, last_error = NULL, updated_at = ?"
                    f" WHERE accession IN ({marks}) AND kind = 'extract' AND status IN ('done', 'failed')"
                    " AND (? IS NULL OR route = ?)", (_now(), *accessions, route, route)).rowcount
        return {"removed": len(stale), "added": added, "requeued": requeued}

    def recover(self, backends: list[str]) -> int:
        """Put items of `backends` left "running" by a crash back to "pending". Call when a runner starts.

        Only the starting runner's own backends are touched: another runner process (e.g. a
        separate Sonnet runner) may be running its own items right now. Returns how many were
        recovered. Their attempt already counted, so an item that crashes the runner every time
        still ends up failed after MAX_ATTEMPTS.
        """
        marks = ",".join("?" * len(backends))
        with self.db:
            cur = self.db.execute(
                "UPDATE items SET status = CASE WHEN attempts >= ? THEN 'failed' ELSE 'pending' END,"
                " last_error = COALESCE(last_error, 'interrupted while running'), updated_at = ?"
                f" WHERE status = 'running' AND backend IN ({marks})", (MAX_ATTEMPTS, _now(), *backends))
        return cur.rowcount

    def has_work(self, backend: str) -> bool:
        """True if `backend` could still do something: a company it owns, or an unclaimed one, has ready items.

        Only looks, never claims. The worker asks this before waiting on a closed gate, so a
        backend with nothing left ends at once instead of waiting (e.g. until midnight) for
        quota it no longer needs, and no company is claimed by a backend that cannot run now.
        """
        has_ready = f"EXISTS (SELECT 1 FROM items i WHERE i.ticker = c.ticker AND {_READY})"
        return self.db.execute(f"SELECT 1 FROM companies c WHERE (backend = ? OR backend IS NULL) AND {has_ready}"
                               " LIMIT 1", (backend, MAX_ATTEMPTS)).fetchone() is not None

    def company_for(self, backend: str) -> str | None:
        """Return the company `backend` should work on, claiming a new one if needed (the taxi rank).

        First, a company the backend already owns that still has ready items (oldest claim
        first). Otherwise the highest-priority unclaimed company with ready items is claimed
        for it, in one SQL statement, so two backends can never claim the same company. Once
        claimed, a company stays with its backend: one model extracts all its years.
        Returns None when there is no work for this backend.
        """
        has_ready = f"EXISTS (SELECT 1 FROM items i WHERE i.ticker = c.ticker AND {_READY})"
        row = self.db.execute(f"SELECT ticker FROM companies c WHERE backend = ? AND {has_ready}"
                              " ORDER BY seq LIMIT 1", (backend, MAX_ATTEMPTS)).fetchone()
        if row:
            return row[0]
        with self.db:
            row = self.db.execute(
                "UPDATE companies SET backend = ?, assigned_at = ? WHERE ticker = ("
                f" SELECT ticker FROM companies c WHERE backend IS NULL AND {has_ready}"
                " ORDER BY seq LIMIT 1) RETURNING ticker", (backend, _now(), MAX_ATTEMPTS)).fetchone()
        return row[0] if row else None

    def ready_tickers(self, route: str) -> list[str]:
        """Companies that have items of this route ready for an LLM call, in ticker order.

        Ready = pending, or failed with attempts left. The batcher uses this to find work for
        the companies assigned to a backend.
        """
        return [r[0] for r in self.db.execute(
            f"SELECT DISTINCT ticker FROM items WHERE route = ? AND {_READY} ORDER BY ticker",
            (route, MAX_ATTEMPTS))]

    def claim(self, route: str, ticker: str, limit: int, backend: str) -> list[sqlite3.Row]:
        """Take up to `limit` ready items of one company and route, and mark them "running".

        Picking and marking happen in one SQL statement, so two workers running at the same
        time (one per backend) can never take the same item. Items come oldest fiscal year
        first, in document order. The attempt is counted here, before the LLM call starts.
        """
        with self.db:
            rows = self.db.execute(
                "UPDATE items SET status = 'running', attempts = attempts + 1, backend = ?, updated_at = ?"
                f" WHERE item_id IN (SELECT item_id FROM items WHERE route = ? AND ticker = ? AND {_READY}"
                "  ORDER BY fiscal_year, chunk_id LIMIT ?) RETURNING *",
                (backend, _now(), route, ticker, MAX_ATTEMPTS, limit)).fetchall()
        # RETURNING gives no guaranteed order: restore oldest-first document order.
        return sorted(rows, key=lambda r: (r["fiscal_year"], r["chunk_id"]))

    def finish(self, done: dict[str, dict], failed: dict[str, str], backend: str, model: str,
               prompt_version: int) -> None:
        """Save the results of one LLM call and update its items, all in one transaction.

        `done` maps item_id -> result payload (saved to `results`, item marked done);
        `failed` maps item_id -> reason (item marked failed, retried while attempts are left).
        Because results and statuses are committed together, a crash can never leave an item
        marked done without its result, or a paid-for result without its item marked done.
        """
        now = _now()
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO results (item_id, payload, backend, model, prompt_version, created_at)"
                " VALUES (?,?,?,?,?,?)",
                [(i, json.dumps(p), backend, model, prompt_version, now) for i, p in done.items()])
            self.db.executemany(
                "UPDATE items SET status = 'done', last_error = NULL, updated_at = ? WHERE item_id = ?",
                [(now, i) for i in done])
            self.db.executemany(
                "UPDATE items SET status = 'failed', last_error = ?, updated_at = ? WHERE item_id = ?",
                [(err[:500], now, i) for i, err in failed.items()])

    def mark_failed(self, item_ids: list[str], error: str) -> None:
        """Mark items as failed with the reason; they are retried while attempts are left."""
        self.finish({}, dict.fromkeys(item_ids, error), "", "", 0)

    def release(self, item_ids: list[str], reason: str) -> None:
        """Put claimed items back to "pending" without counting the attempt.

        Used when a call did not happen or failed because of the backend's quota, not because
        of the items: such a failure must not use up the items' MAX_ATTEMPTS.
        """
        with self.db:
            self.db.executemany(
                "UPDATE items SET status = 'pending', attempts = MAX(attempts - 1, 0), last_error = ?,"
                " updated_at = ? WHERE item_id = ?", [(reason[:500], _now(), i) for i in item_ids])

    def log_call(self, backend: str, route: str, ticker: str, n_items: int, n_done: int,
                 input_tokens: int, output_tokens: int, seconds: float, cached: bool,
                 error: str | None = None) -> None:
        """Record one LLM call (tokens, time, outcome); progress reports and ETAs are computed from these."""
        with self.db:
            self.db.execute(
                "INSERT INTO calls (backend, route, ticker, n_items, n_done, input_tokens, output_tokens,"
                " seconds, cached, error, at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (backend, route, ticker, n_items, n_done, input_tokens, output_tokens, seconds,
                 int(cached), error and error[:500], _now()))

    def counts(self) -> list[sqlite3.Row]:
        """Number of items and their tokens per company, route and status (for status reports)."""
        return self.db.execute(
            "SELECT ticker, route, status, COUNT(*) AS n, SUM(tokens) AS tokens FROM items"
            " GROUP BY ticker, route, status ORDER BY ticker, route, status").fetchall()


def print_status(store: JobStore) -> None:
    """Print a table of item counts per company and route, by status, plus a total line."""
    statuses = ["pending", "running", "done", "failed", "reuse"]
    table: dict[tuple[str, str], dict[str, int]] = {}
    for r in store.counts():
        for key in ((r["ticker"], r["route"]), ("TOTAL", r["route"])):
            table.setdefault(key, dict.fromkeys(statuses, 0))[r["status"]] += r["n"]
    print(f"{'company':8}{'route':>6}" + "".join(f"{s:>9}" for s in statuses))
    for (ticker, route), row in sorted(table.items(), key=lambda kv: (kv[0][0] == "TOTAL", kv[0])):
        print(f"{ticker:8}{route:>6}" + "".join(f"{row[s]:>9}" for s in statuses))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Manage the extraction job store.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_add = sub.add_parser("add", help="plan a wave of companies and add its items")
    p_add.add_argument("tickers", nargs="+")
    p_add.add_argument("--first-years", type=int, help="only each company's first N filings")
    p_re = sub.add_parser("replan", help="re-plan filings after a prompt/schema change (see JobStore.replan)")
    p_re.add_argument("tickers", nargs="+")
    p_re.add_argument("--first-years", type=int, help="only each company's first N filings")
    p_re.add_argument("--requeue", action="store_true", help="also send done/failed items to the LLM again")
    p_re.add_argument("--route", choices=["A", "B"], help="requeue only this route")
    sub.add_parser("status", help="print item counts by status")
    ap.add_argument("--db", type=Path, default=JOBS_DB, help="job store file (default: the real one)")
    a = ap.parse_args()
    js = JobStore(a.db)
    if a.cmd == "add":
        print(f"added {js.add(plan_wave(a.tickers, a.first_years))} new items")
    elif a.cmd == "replan":
        print(js.replan(plan_wave(a.tickers, a.first_years), a.requeue, a.route))
    print_status(js)
