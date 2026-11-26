"""Stage 5 of the extraction pipeline: make the LLM calls, check the answers, record the outcome.

What it does: one worker loop runs per backend (each in its own thread). Each round it
  1. asks the router whether the backend may run now (if not, sleeps as long as told),
  2. gets the backend's company from the router (the taxi rank),
  3. takes the next batch of that company from the batcher (route A first, then route B),
  4. sends it through the LLM cache, validates the answer, and saves results and statuses
     together in one transaction (`JobStore.finish`).
How each outcome is handled:
  - good answer: every item with a usable answer is done; an item the answer skipped or got
    wrong is failed and is retried later in another batch
  - answer is not valid JSON: the cached answer is deleted (otherwise a retry would get the
    same bad answer back from the cache), and the items are failed for a retry
  - the call itself fails: the meters are re-read; if the gate is now closed, the failure was
    the quota running out, so the items go back to pending without using an attempt;
    otherwise they are failed and the worker backs off (30 s, doubling, at most 15 min)
  - the daily budget refuses the call: the items go back to pending, the worker waits
A loop ends when its backend has no work left (checked before the gate, so a backend never
waits for quota it no longer needs), after `max_batches`, when stopped, or, in stop-when-blocked
mode, as soon as its limits block it (the user restarts the run once they are restored).
Along the way it writes progress events (see `progress.py`): gate closed/reopened, Sonnet
needs an allowance, company finished, errors in a row, no work left.

Why it is needed: this is where quota is actually spent, so it is where "never pay twice" and
"a crash loses nothing" must hold: answers are cached before they are used, and an item is
only marked done in the same transaction that saves its result.

Run (starts real LLM calls; only through the user-invoked run skills):
  .venv/bin/python -m sec10k.extraction_pipeline.worker --backends groq antigravity_cli \
      [--max-batches N] [--stop-when-blocked]
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sec10k.config import EXTRACTION_LOG, JOBS_DB
from sec10k.extract.llm_extract import (PROMPT_VERSION, _filer_name, answers_a, answers_b, call_cached,
                                        drop_cached)
from sec10k.extraction_pipeline import progress
from sec10k.extraction_pipeline.batcher import Batch, Batcher
from sec10k.extraction_pipeline.jobs import JobStore
from sec10k.extraction_pipeline.router import NO_ALLOWANCE_WAIT, MeterReader, Router
from sec10k.llm.providers import BudgetExceeded, LLMProvider, get_llm

log = logging.getLogger("sec10k.extraction")

BACKOFF_START = 30     # seconds to wait after the first failed call in a row
BACKOFF_MAX = 900      # the wait doubles with each failed call in a row, up to this


@dataclass
class BatchResult:
    """What happened to one batch: the outcome (see `process_batch`), items done, tokens, and cache use."""

    outcome: str
    n_done: int = 0
    tokens: int = 0
    cached: bool = False


def process_batch(store: JobStore, router: Router, llm: LLMProvider, batch: Batch) -> BatchResult:
    """Send one batch to the LLM, validate the answer and record everything.

    Outcomes: "ok", "bad_json", "error" (call failed, items failed), "quota" (call failed
    because the quota ran out, items released), "budget" (call refused by the daily budget,
    items released). The module docstring explains how each is handled.
    """
    ids = [it["item_id"] for it in batch.items]
    try:
        res = call_cached(llm, batch.system, batch.prompt, batch.max_tokens)
    except BudgetExceeded as e:
        store.release(ids, f"budget: {e}")
        return BatchResult("budget")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        router.meters.invalidate(llm.backend)
        quota = type(e).__name__ == "RateLimitError" or not router.gate(llm).allowed
        store.log_call(llm.backend, batch.route, batch.ticker, len(ids), 0, 0, 0, 0.0, False, err)
        if quota:
            store.release(ids, f"quota: {err}")
            return BatchResult("quota")
        store.mark_failed(ids, err)
        return BatchResult("error")

    r = res.response
    tokens = r.input_tokens + r.output_tokens
    if res.data is None:
        drop_cached(llm, batch.system, batch.prompt)
        store.finish({}, dict.fromkeys(ids, "answer was not valid JSON"), llm.backend, r.model, PROMPT_VERSION)
        store.log_call(llm.backend, batch.route, batch.ticker, len(ids), 0, r.input_tokens, r.output_tokens,
                       res.seconds, res.cached, "answer was not valid JSON")
        return BatchResult("bad_json", 0, tokens, res.cached)

    stats: Counter = Counter()
    if batch.route == "A":
        payloads = answers_a(res.data, [it["title"] for it in batch.items], stats)
    else:
        facts = answers_b(res.data, batch.chunks, _filer_name(batch.ticker), stats)
        payloads = [None if f is None else {"facts": f} for f in facts]
    done = {it["item_id"]: p for it, p in zip(batch.items, payloads) if p is not None}
    failed = {it["item_id"]: "missing or invalid in the answer" for it, p in zip(batch.items, payloads) if p is None}
    store.finish(done, failed, llm.backend, r.model, PROMPT_VERSION)
    store.log_call(llm.backend, batch.route, batch.ticker, len(ids), len(done), r.input_tokens, r.output_tokens,
                   res.seconds, res.cached)
    return BatchResult("ok", len(done), tokens, res.cached)


def _company_counts(store: JobStore, ticker: str) -> str:
    """One-line summary of a company's extract items by status, for the "company finished" event."""
    rows = store.db.execute("SELECT status, COUNT(*) FROM items WHERE ticker = ? AND kind = 'extract'"
                            " GROUP BY status", (ticker,)).fetchall()
    return ", ".join(f"{n} {s}" for s, n in rows)


def run_backend(backend: str, stop: threading.Event, meters: MeterReader, db_path: Path = JOBS_DB,
                max_batches: int | None = None, llm: LLMProvider | None = None,
                stop_when_blocked: bool = False) -> Counter:
    """The worker loop of one backend (see the module docstring). Returns a count of outcomes.

    Each worker opens its own job store connection (SQLite connections are not shared between
    threads); `meters` is shared, so the usage meters are read once for all workers.
    `llm` can be given in tests to use a fake backend.
    With `stop_when_blocked`, the loop ends (instead of waiting) as soon as the backend is
    blocked: its gate is closed, or a call is refused by the budget or fails on the quota.
    The user then restarts the run once the limits are restored; nothing is lost, since the
    remaining items stay pending and each company keeps its backend.
    """
    store = JobStore(db_path)
    router = Router(store, meters)
    llm = llm or get_llm(backend)
    batcher = Batcher(store, backend)
    outcomes: Counter = Counter()
    backoff, errors_in_row = 0, 0
    gate_was_open: bool | None = None       # None until the first check, so the first state is reported
    last_allowance_ask = 0.0
    company: str | None = None

    def no_work_left() -> None:
        """Report the last company as finished (if any) and the backend as idle."""
        if company is not None:
            progress.emit("company_done", f"{company} finished: {_company_counts(store, company)}",
                          push=True, backend=backend, ticker=company)
        progress.emit("backend_idle", "no work left", push=True, backend=backend)
        log.info("%s: no work left", backend)

    def blocked(reason: str) -> None:
        """Report that the backend stops because it is blocked (stop-when-blocked mode)."""
        progress.emit("backend_stopped", f"stopped, blocked: {reason}. Restart with /extract-run once the "
                      "limit is restored", push=True, backend=backend)
        log.info("%s: stopped, blocked (%s)", backend, reason)

    while not stop.is_set():
        if max_batches is not None and sum(outcomes.values()) >= max_batches:
            log.info("%s: stopping after %d batches", backend, max_batches)
            break
        # Check for work before the gate: a backend with nothing left must end now, not wait
        # (e.g. until midnight) for quota it no longer needs.
        if not store.has_work(backend):
            no_work_left()
            break
        gate = router.gate(llm)
        if not gate.allowed and stop_when_blocked:
            blocked(gate.reason)
            break
        if gate.allowed != gate_was_open:
            kind = "gate_opened" if gate.allowed else "gate_closed"
            progress.emit(kind, gate.reason, push=gate_was_open is not None or not gate.allowed, backend=backend,
                          wait_minutes=gate.wait_seconds // 60)
            gate_was_open = gate.allowed
        if not gate.allowed:
            if gate.needs_allowance and time.monotonic() - last_allowance_ask > progress.ALLOWANCE_REMINDER_MIN * 60:
                progress.emit("needs_allowance", f"Sonnet waiting: {gate.reason}. Reply with /extract-sonnet <5-hour %>",
                              push=True, backend=backend)
                last_allowance_ask = time.monotonic()
            log.info("%s: waiting %d min (%s)", backend, gate.wait_seconds // 60, gate.reason)
            stop.wait(gate.wait_seconds)
            continue
        ticker = router.next_company(backend)
        if ticker is None:          # another backend took the last company since has_work()
            no_work_left()
            break
        if ticker != company and company is not None:
            progress.emit("company_done", f"{company} finished: {_company_counts(store, company)}",
                          push=True, backend=backend, ticker=company)
        company = ticker
        batch = batcher.next_batch("A", [ticker]) or batcher.next_batch("B", [ticker])
        if batch is None:
            stop.wait(1)
            continue
        res = process_batch(store, router, llm, batch)
        outcomes[res.outcome] += 1
        log.info("%s: %s route %s, %d items -> %s (%d done, %s tokens%s)", backend, batch.ticker, batch.route,
                 len(batch.items), res.outcome, res.n_done, f"{res.tokens:,}", ", cached" if res.cached else "")
        errors_in_row = errors_in_row + 1 if res.outcome == "error" else 0
        if errors_in_row == progress.ERRORS_IN_A_ROW:
            progress.emit("errors", f"{errors_in_row} failed calls in a row; last on {batch.ticker}, backing off",
                          push=True, backend=backend)
        if res.outcome == "error":
            backoff = min(backoff * 2 or BACKOFF_START, BACKOFF_MAX)
            stop.wait(backoff)
        elif res.outcome in ("budget", "quota"):
            if stop_when_blocked:
                blocked(f"call refused or failed on the {res.outcome}")
                break
            stop.wait(NO_ALLOWANCE_WAIT)
        else:
            backoff = 0
    return outcomes


def run(backends: list[str], db_path: Path = JOBS_DB, max_batches: int | None = None,
        stop: threading.Event | None = None, stop_when_blocked: bool = False) -> None:
    """Run one worker thread per backend until all are done or `stop` is set.

    First puts this runner's backends' items that a previous crash left "running" back to
    pending (`JobStore.recover`); other runner processes' items are left alone.
    Emits a start event, a summary every SUMMARY_EVERY_MIN minutes, and a stop event with
    the final summary. With `stop_when_blocked` each backend ends when it is blocked instead
    of waiting for its limit to reset (see `run_backend`). `stop` is set by the signal handlers (`_stop_on_signals`); once set,
    each worker finishes its call in progress and ends. The main thread waits for each
    worker's own "finished" flag rather than an interruptible `Thread.join()`, so the calls
    in progress always complete before the process exits.
    """
    store = JobStore(db_path)
    recovered = store.recover(backends)
    progress.emit("run_started", f"backends {', '.join(backends)}; recovered {recovered} interrupted items"
                  + (f"; at most {max_batches} batches each" if max_batches else "")
                  + ("; stops when blocked" if stop_when_blocked else ""), push=True)
    stop, meters = stop or threading.Event(), MeterReader()
    finished = {b: threading.Event() for b in backends}

    def worker(b: str) -> None:
        """One backend's loop; always signals "finished", even if the loop fails."""
        try:
            run_backend(b, stop, meters, db_path, max_batches, stop_when_blocked=stop_when_blocked)
        except Exception:
            log.exception("%s: worker failed", b)
            progress.emit("errors", f"worker crashed: see {EXTRACTION_LOG}", push=True, backend=b)
        finally:
            finished[b].set()

    threads = [threading.Thread(target=worker, args=(b,), name=b) for b in backends]
    for t in threads:
        t.start()
    next_summary = time.monotonic() + progress.SUMMARY_EVERY_MIN * 60
    while not all(f.is_set() for f in finished.values()):
        for f in finished.values():
            f.wait(timeout=1)
        if time.monotonic() >= next_summary:
            progress.emit("summary", progress.status_report(store), push=True)
            next_summary += progress.SUMMARY_EVERY_MIN * 60
    for t in threads:
        t.join()          # all workers have finished, so this returns at once
    progress.emit("run_stopped", progress.status_report(store), push=True)


def _stop_on_signals(stop: threading.Event) -> None:
    """Make SIGINT and SIGTERM stop the runner gracefully (as Ctrl-C does in a terminal).

    A process started in the background by a script (`nohup … &`, as the run skills do)
    inherits SIGINT as "ignored", so without this `kill -INT` (what /wind-down sends) would
    do nothing. The handler only sets `stop`; it raises no exception, because an exception
    that interrupts the main thread's waiting can make Python exit while a worker is still
    in a call. The workers see `stop`, finish their call in progress and end; `run()` then
    returns. Repeated signals (e.g. `caffeinate` forwards the one it gets) change nothing.
    """
    def handler(signum, frame):
        if not stop.is_set():
            log.info("stop requested (signal %d): finishing the calls in progress", signum)
        stop.set()
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run extraction workers (spends LLM quota).")
    ap.add_argument("--backends", nargs="+", required=True, choices=["groq", "antigravity_cli", "claude_cli"])
    ap.add_argument("--max-batches", type=int, help="stop each backend after this many batches")
    ap.add_argument("--stop-when-blocked", action="store_true",
                    help="end each backend when its limits block it, instead of waiting for them to reset")
    a = ap.parse_args()
    EXTRACTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%m-%d %H:%M:%S",
                        handlers=[logging.StreamHandler(), logging.FileHandler(EXTRACTION_LOG)])
    stop_event = threading.Event()
    _stop_on_signals(stop_event)
    run(a.backends, max_batches=a.max_batches, stop=stop_event, stop_when_blocked=a.stop_when_blocked)
