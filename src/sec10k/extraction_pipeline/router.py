"""Stage 4 of the extraction pipeline: decide whether a backend may run now, and on which company.

What it does, before every batch of a backend's worker:
  1. Gate: "may this backend make a call right now?" Each backend has its own rules:
     - groq:            its daily token budget (the per-minute limits are handled inside the
                        provider, which simply waits)
     - antigravity_cli: the `agy -p /usage` meters for Gemini models: pause while the 5-hour
                        meter is at >= 95% (until it resets), stop while the weekly one is >= 90%
     - claude_cli:      the Sonnet schedule, read from `claude -p /usage`:
                        * night (23:00-07:00 IST) and weekly < 75%: runs, up to 95% of the 5-hour window
                        * otherwise only with the user's allowance (`allow-sonnet`), up to the
                          allowed 5-hour %, and never at weekly >= 90% unless the allowance
                          explicitly overrides that
     A closed gate says why and how long to wait before asking again; a missing allowance is
     flagged so the progress stage can ask the user for one by phone push.
  2. Company: the taxi rank. A backend keeps working on the company it owns; when that is
     done it claims the next unclaimed company in priority order (see `JobStore.company_for`),
     so one model extracts all years of a company and faster backends take more companies.

Why it is needed: the three backends run in parallel on separate quotas. Without these gates
a run would hit quota errors, eat into the Claude Pro plan the user needs during the day, or
spend Antigravity's weekly quota in one go.

Meter readings are cached for a few minutes, because `agy -p /usage` takes ~10 s. If a meter
cannot be read, the gate stays closed (never spend quota blind).

Run:
  .venv/bin/python -m sec10k.extraction_pipeline.router status
  .venv/bin/python -m sec10k.extraction_pipeline.router allow-sonnet 60 [--over-weekly]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sec10k.config import SONNET_ALLOWANCE
from sec10k.extraction_pipeline.jobs import JobStore

IST = ZoneInfo("Asia/Kolkata")
NIGHT_START, NIGHT_END = 23, 7      # Sonnet's night window, hours in IST
SONNET_SESSION_MAX = 95             # % of the 5-hour window Sonnet may use at night
SONNET_NIGHT_WEEK_MAX = 75          # night runs stop once weekly Claude usage reaches this %
SONNET_DAY_WEEK_MAX = 90            # no Sonnet at all from this weekly % on, unless explicitly overridden
AGY_SESSION_MAX = 95                # Antigravity pauses at this % of its 5-hour quota
AGY_WEEK_MAX = 90                   # Antigravity stops for the week at this % of its weekly quota
METER_TTL = 180                     # seconds a meter reading is reused
MAX_WAIT = 1800                     # a closed gate is checked again after at most this many seconds
NO_ALLOWANCE_WAIT = 300             # how often a worker waiting for an allowance checks again
# Rough tokens of one batch (prompt + allowed output), to check a daily budget before claiming work.
BATCH_ESTIMATE = {"groq": 6_000}
BATCH_ESTIMATE_DEFAULT = 12_000


# --------------------------------------------------------------------------------------
# Usage meters
# --------------------------------------------------------------------------------------
@dataclass
class Meters:
    """One reading of a backend's quota meters: % used of the 5-hour and weekly windows, and when they reset."""

    session_used: float
    week_used: float
    session_resets: datetime | None
    week_resets: datetime | None


def _parse_claude_reset(text: str, now: datetime) -> datetime | None:
    """Parse Claude's reset text, e.g. "resets Nov 26 at 7:50pm (Asia/Calcutta)" or "resets 8pm (...)".

    Returns None if the text has an unexpected shape (the gate then simply checks again later).
    """
    m = re.search(r"resets (?:(\w{3} \d{1,2}) at )?(\d{1,2})(?::(\d{2}))?\s*([ap]m)\s*\(([^)]+)\)", text)
    if not m:
        return None
    day, hour, minute, ampm, tz = m.groups()
    tz = ZoneInfo(tz)
    local_now = now.astimezone(tz)
    hour = int(hour) % 12 + (12 if ampm == "pm" else 0)
    if day:
        d = datetime.strptime(f"{day} {local_now.year}", "%b %d %Y").date()
    else:
        d = local_now.date()
    out = datetime(d.year, d.month, d.day, hour, int(minute or 0), tzinfo=tz)
    if day and out < local_now - timedelta(days=180):     # "Jan 2" read in late December
        out = out.replace(year=out.year + 1)
    if not day and out < local_now:                         # time only: the next occurrence
        out += timedelta(days=1)
    return out


def parse_claude_usage(text: str, now: datetime) -> Meters:
    """Read the session and weekly % from `claude -p /usage` output."""
    session = re.search(r"Current session:\s*(\d+(?:\.\d+)?)% used(.*)", text)
    week = re.search(r"Current week \(all models\):\s*(\d+(?:\.\d+)?)% used(.*)", text)
    if not session or not week:
        raise ValueError(f"unexpected `claude -p /usage` output: {text[:300]!r}")
    return Meters(float(session.group(1)), float(week.group(1)),
                  _parse_claude_reset(session.group(2), now), _parse_claude_reset(week.group(2), now))


def parse_agy_usage(text: str) -> Meters:
    """Read the Gemini 5-hour and weekly meters from `agy -p /usage` output (tab-separated lines).

    `agy` reports the % *remaining*; this returns % used, like the Claude meters.
    """
    found = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) == 4 and parts[0] == "Gemini Models":
            used = 100 - float(parts[2].rstrip("%"))
            found[parts[1]] = (used, datetime.fromisoformat(parts[3]))
    if "Weekly Limit Remaining" not in found or "Five Hour Limit Remaining" not in found:
        raise ValueError(f"unexpected `agy -p /usage` output: {text[:300]!r}")
    (s_used, s_reset), (w_used, w_reset) = found["Five Hour Limit Remaining"], found["Weekly Limit Remaining"]
    return Meters(s_used, w_used, s_reset, w_reset)


class MeterReader:
    """Reads the Claude and Antigravity usage meters, reusing a reading for METER_TTL seconds.

    Shared by the worker threads; a lock keeps two threads from reading the same meter at once.
    """

    COMMANDS = {"claude_cli": ["claude", "-p", "/usage"], "antigravity_cli": ["agy", "-p", "/usage"]}

    def __init__(self):
        """Start with an empty cache."""
        self._cache: dict[str, tuple[float, Meters]] = {}
        self._lock = threading.Lock()

    def get(self, backend: str) -> Meters:
        """Return the backend's meters, from the cache if fresh, else by running its /usage command."""
        with self._lock:
            hit = self._cache.get(backend)
            if hit and time.monotonic() - hit[0] < METER_TTL:
                return hit[1]
            with tempfile.TemporaryDirectory() as cwd:   # empty folder: no project files picked up
                out = subprocess.run(self.COMMANDS[backend], capture_output=True, text=True,
                                     cwd=cwd, timeout=120).stdout
            meters = (parse_claude_usage(out, datetime.now(IST)) if backend == "claude_cli"
                      else parse_agy_usage(out))
            self._cache[backend] = (time.monotonic(), meters)
            return meters

    def invalidate(self, backend: str) -> None:
        """Forget the cached reading, so the next `get` reads the meter fresh.

        The worker calls this after a failed call, to find out whether the quota ran out.
        """
        with self._lock:
            self._cache.pop(backend, None)


# --------------------------------------------------------------------------------------
# The user's daytime Sonnet allowance
# --------------------------------------------------------------------------------------
def grant_sonnet_allowance(max_session_pct: float, over_weekly: bool, meters: Meters, now: datetime) -> dict:
    """Save the user's allowance: Sonnet may run until the 5-hour meter reaches `max_session_pct`.

    It is valid until the current 5-hour window resets (or 5 hours if the reset time is
    unknown); after that a new allowance is needed. `over_weekly` lets Sonnet run even at
    weekly usage >= SONNET_DAY_WEEK_MAX, which the user must ask for explicitly.
    Called by the `/extract-run sonnet <PCT>` skill through this module's command line.
    """
    expires = meters.session_resets or now + timedelta(hours=5)
    data = {"max_session_pct": max_session_pct, "over_weekly": over_weekly,
            "granted_at": now.isoformat(timespec="seconds"), "expires_at": expires.isoformat(timespec="seconds")}
    SONNET_ALLOWANCE.parent.mkdir(parents=True, exist_ok=True)
    SONNET_ALLOWANCE.write_text(json.dumps(data, indent=1))
    return data


def load_sonnet_allowance(now: datetime) -> dict | None:
    """Return the current allowance, or None if there is none or it has expired."""
    if not SONNET_ALLOWANCE.exists():
        return None
    data = json.loads(SONNET_ALLOWANCE.read_text())
    return data if datetime.fromisoformat(data["expires_at"]) > now else None


# --------------------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------------------
@dataclass
class Gate:
    """The router's answer for one backend: may it run, why (or why not), and when to ask again.

    `needs_allowance` is True when only a user allowance can open the gate, so the progress
    stage knows to send the phone push.
    """

    allowed: bool
    reason: str
    wait_seconds: int = 0
    needs_allowance: bool = False


def _wait_until(when: datetime | None, now: datetime) -> int:
    """Seconds to wait before checking again: until `when` (plus a minute), but at most MAX_WAIT."""
    if when is None:
        return MAX_WAIT
    return int(min(max((when - now).total_seconds() + 60, 60), MAX_WAIT))


def is_night(now: datetime) -> bool:
    """True during Sonnet's night window (23:00-07:00 IST)."""
    h = now.astimezone(IST).hour
    return h >= NIGHT_START or h < NIGHT_END


class Router:
    """Opens or closes each backend's gate and hands out companies (see the module docstring).

    `meters` and `clock` can be replaced in tests, so every rule can be checked without
    reading real meters or waiting for night.
    """

    def __init__(self, store: JobStore | None, meters: MeterReader | None = None, clock=None):
        """Create a router over the job store, with a meter reader and a clock (IST now by default).

        `store` may be None when only gates are needed (the `status` command), so checking the
        gates never creates the job store file.
        """
        self.store = store
        self.meters = meters or MeterReader()
        self.clock = clock or (lambda: datetime.now(IST))

    def gate(self, llm) -> Gate:
        """Decide whether backend `llm` (an LLMProvider) may make its next call now.

        The daily token budget is checked for every backend; then the backend's own rules.
        """
        now = self.clock()
        estimate = BATCH_ESTIMATE.get(llm.backend, BATCH_ESTIMATE_DEFAULT)
        if llm.budget.used_today() + estimate > llm.budget.daily_limit:
            midnight = (now.astimezone() + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            return Gate(False, f"daily token budget used ({llm.budget.used_today():,} / "
                               f"{llm.budget.daily_limit:,})", _wait_until(midnight, now))
        if llm.backend == "groq":
            return Gate(True, "within daily budget")
        try:
            m = self.meters.get(llm.backend)
        except Exception as e:   # unreadable meter: never spend quota blind
            return Gate(False, f"usage meter unreadable: {type(e).__name__}: {e}"[:300], NO_ALLOWANCE_WAIT)
        if llm.backend == "antigravity_cli":
            return self._gate_antigravity(m, now)
        if llm.backend == "claude_cli":
            return self._gate_sonnet(m, now)
        return Gate(False, f"no rules for backend {llm.backend!r}", MAX_WAIT)

    def _gate_antigravity(self, m: Meters, now: datetime) -> Gate:
        """Antigravity: stop at weekly >= AGY_WEEK_MAX, pause at 5-hour >= AGY_SESSION_MAX."""
        if m.week_used >= AGY_WEEK_MAX:
            return Gate(False, f"weekly {m.week_used:.0f}% >= {AGY_WEEK_MAX}%", _wait_until(m.week_resets, now))
        if m.session_used >= AGY_SESSION_MAX:
            return Gate(False, f"5-hour {m.session_used:.0f}% >= {AGY_SESSION_MAX}%", _wait_until(m.session_resets, now))
        return Gate(True, f"5-hour {m.session_used:.0f}%, weekly {m.week_used:.0f}%")

    def _gate_sonnet(self, m: Meters, now: datetime) -> Gate:
        """Sonnet: the night rule first, otherwise the user's allowance (see the module docstring)."""
        meters = f"5-hour {m.session_used:.0f}%, weekly {m.week_used:.0f}%"
        if m.session_used >= SONNET_SESSION_MAX:
            return Gate(False, f"{meters}: 5-hour window full", _wait_until(m.session_resets, now))
        if is_night(now) and m.week_used < SONNET_NIGHT_WEEK_MAX:
            return Gate(True, f"night, {meters}")
        allowance = load_sonnet_allowance(now)
        when = "night but weekly >= 75%" if is_night(now) else "daytime"
        if allowance is None:
            return Gate(False, f"{when}, {meters}: waiting for an allowance", NO_ALLOWANCE_WAIT, needs_allowance=True)
        if m.week_used >= SONNET_DAY_WEEK_MAX and not allowance["over_weekly"]:
            return Gate(False, f"{meters}: weekly >= {SONNET_DAY_WEEK_MAX}%, needs an explicit override",
                        NO_ALLOWANCE_WAIT, needs_allowance=True)
        if m.session_used >= allowance["max_session_pct"]:
            return Gate(False, f"{meters}: allowance of {allowance['max_session_pct']:.0f}% used up",
                        _wait_until(m.session_resets, now))
        return Gate(True, f"{when}, allowance up to {allowance['max_session_pct']:.0f}%, {meters}")

    def next_company(self, backend: str) -> str | None:
        """The company this backend should work on next (the taxi rank), or None if there is no work."""
        return self.store.company_for(backend)


if __name__ == "__main__":
    from sec10k.llm.providers import get_llm

    ap = argparse.ArgumentParser(description="Show backend gates, or grant Sonnet a daytime allowance.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="show each backend's gate right now (reads the usage meters)")
    p = sub.add_parser("allow-sonnet", help="let Sonnet run until the 5-hour meter reaches PCT")
    p.add_argument("pct", type=float)
    p.add_argument("--over-weekly", action="store_true", help=f"also allow at weekly >= {SONNET_DAY_WEEK_MAX}%%")
    a = ap.parse_args()
    if a.cmd == "allow-sonnet":
        now = datetime.now(IST)
        print(grant_sonnet_allowance(a.pct, a.over_weekly, MeterReader().get("claude_cli"), now))
    else:
        router = Router(None)
        for name in ("groq", "antigravity_cli", "claude_cli"):
            g = router.gate(get_llm(name))
            print(f"{name:16} {'OPEN  ' if g.allowed else 'CLOSED'} {g.reason}"
                  + (f" (check again in {g.wait_seconds // 60} min)" if not g.allowed else "")
                  + (" [needs allowance]" if g.needs_allowance else ""))
