"""One common way to call an LLM, with swappable backends.

What it does: defines `LLMProvider`, a class with a single `complete(prompt)` call, and three
backends that implement it:
  - GroqProvider            the "real" production path (openai/gpt-oss-120b over Groq's
                            OpenAI-compatible API, free tier)
  - ClaudeCLIProvider       a WORKAROUND for the free tier's small limits: it runs `claude -p`
                            (Claude Code headless, on the Pro plan)
  - AntigravityCLIProvider  a second WORKAROUND with its own separate quota: it runs `agy`
                            (Google Antigravity headless) with a Gemini model
Which backend is used is set by the LLM_BACKEND environment variable (see `get_llm()`).

Why it is needed: the rest of the project only talks to `LLMProvider`, so it does not depend
on one vendor, and moving to a different LLM needs no code change outside this file. Entity
extraction (Phase 3) and answer generation will call the LLM through here.

Built-in guards (a simple version; a fuller rate limiter and a response cache are planned
for the production ingestion pipeline):
  - retries with waiting  -> the openai SDK's built-in `max_retries` (Groq only)
  - request pacing        -> at most N requests per minute, per backend
  - tokens per minute     -> waits until the last 60 seconds have room for the next call
                             (only for backends with a per-minute token limit, i.e. Groq)
  - daily token budget    -> refuses to call once today's tokens would pass the limit; the
                             count is saved to a JSON file so it survives restarts. Groq's
                             budget is a bucket that refills steadily instead of a calendar-day
                             count, because that is how Groq itself applies its daily limit
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from sec10k.config import STATE_DIR  # also loads .env


# --------------------------------------------------------------------------------------
# Result type: every backend returns the same shape, so callers never care which one answered
# --------------------------------------------------------------------------------------
@dataclass
class LLMResponse:
    """The answer of one LLM call: the text, the tokens used, and which model/backend gave it."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str
    backend: str

    @property
    def total_tokens(self) -> int:
        """Input plus output tokens; this is what counts against the daily budget."""
        return self.input_tokens + self.output_tokens


class BudgetExceeded(RuntimeError):
    """Raised instead of calling the LLM when today's token budget is used up."""


# --------------------------------------------------------------------------------------
# Guards: a simple daily token budget and request pacing
# --------------------------------------------------------------------------------------
class DailyTokenBudget:
    """Keeps count of the tokens used today by one backend, and blocks calls past the daily limit.

    The count is saved in a small JSON file, so it survives restarts. This protects the free
    daily quota: a crash or a rerun can never use up more tokens than allowed.

    Quotas belong to an account (an API key), not to a backend. With `account` (a short
    fingerprint of the key, never the key itself), `budget_<backend>.json` always holds the
    active account's count and names its account. When the key changes, the other account's
    count is moved aside to `budget_<backend>_<account>.json` and the active account's own
    earlier count, if any, is restored. So switching keys never mixes two accounts' usage,
    and tools that read `budget_<backend>.json` keep working.
    """

    period = "today"  # how messages describe the counting period

    def __init__(self, backend: str, daily_limit: int, account: str | None = None):
        """Create the counter for `backend`, optionally for one account (key fingerprint)."""
        self.backend = backend
        self.daily_limit = daily_limit
        self.account = account
        self.path = STATE_DIR / f"budget_{backend}.json"

    def _fresh(self) -> dict:
        """A zero count for today (for this account, if accounts are tracked)."""
        data = {"date": date.today().isoformat(), "tokens_used": 0, "requests": 0}
        return {**data, "account": self.account} if self.account else data

    def _switch_account(self, data: dict) -> dict:
        """The file holds another account's count: park it, and return this account's own saved count."""
        parked = STATE_DIR / f"budget_{self.backend}_{data.get('account') or 'unknown'}.json"
        parked.write_text(json.dumps(data))
        mine = STATE_DIR / f"budget_{self.backend}_{self.account}.json"
        own = json.loads(mine.read_text()) if mine.exists() else self._fresh()
        self._save(own)
        return own

    def _save(self, data: dict) -> None:
        """Write the counters atomically (the replace means the file is never half-written)."""
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(self.path)

    def _load(self) -> dict:
        """Read today's counters from the file; start from zero on a new day or a missing file."""
        if self.path.exists():
            data = json.loads(self.path.read_text())
            if self.account and data.get("account") != self.account:
                data = self._switch_account(data)
            if data.get("date") == date.today().isoformat():
                return data
        return self._fresh()

    def used_today(self) -> int:
        """Return how many tokens this backend has used today."""
        return self._load()["tokens_used"]

    def check(self, estimated_tokens: int) -> None:
        """Raise `BudgetExceeded` if a call of about `estimated_tokens` would pass the daily limit."""
        used = self.used_today()
        if used + estimated_tokens > self.daily_limit:
            raise BudgetExceeded(
                f"{self.backend}: {used:,} tokens used {self.period} + ~{estimated_tokens:,} "
                f"estimated > daily limit {self.daily_limit:,}"
            )

    def record(self, tokens: int) -> None:
        """Add the tokens of a finished call to today's count and save it."""
        data = self._load()
        data["tokens_used"] += tokens
        data["requests"] += 1
        self._save(data)


class TokenBucketBudget(DailyTokenBudget):
    """A daily token budget that refills steadily, the way Groq's tokens-per-day limit works.

    Think of a bucket that holds the daily limit (200K tokens for Groq's free tier). Each call
    takes its tokens out; the bucket refills at the daily limit per 24 hours (about 8.3K an
    hour), never above full. Groq applies its limit this way, so a counter that restarts at
    midnight, or one that waits for calls to be a full day old, would disagree with Groq:
    the first lets calls through that Groq refuses, the second blocks calls Groq would accept.

    The file stores the bucket's level and when it was last updated; the current level is
    worked out from the time passed since. `used_today()` reports the empty part of the bucket,
    so the router and the providers use it exactly like a `DailyTokenBudget`.
    """

    REFILL_SECONDS = 24 * 3600.0  # time to refill an empty bucket completely
    period = "(refilling bucket)"

    def _fresh(self) -> dict:
        """A full bucket (for this account, if accounts are tracked)."""
        data = {"level": float(self.daily_limit), "updated": time.time()}
        return {**data, "account": self.account} if self.account else data

    def _load(self) -> dict:
        """Read the bucket and refill it for the time passed since it was last saved.

        A file in another format (such as a calendar-day count) starts as a full bucket.
        """
        if self.path.exists():
            data = json.loads(self.path.read_text())
            if self.account and data.get("account") != self.account:
                data = self._switch_account(data)
            if "level" in data:
                now = time.time()
                refill = (now - data["updated"]) * self.daily_limit / self.REFILL_SECONDS
                data["level"] = min(float(self.daily_limit), data["level"] + refill)
                data["updated"] = now
                return data
        return self._fresh()

    def used_today(self) -> int:
        """Return the tokens missing from a full bucket (what Groq would call "used")."""
        return round(self.daily_limit - self._load()["level"])

    def record(self, tokens: int) -> None:
        """Take a finished call's tokens out of the bucket and save it.

        The level may go slightly below zero when a call used more than estimated; it then
        refills from there, as Groq's own count does.
        """
        data = self._load()
        data["level"] -= tokens
        self._save(data)


class RequestPacer:
    """Spaces out requests so a backend never gets more than `per_minute` requests a minute.

    The free API rejects requests above its per-minute limit, so we wait before sending.
    """

    def __init__(self, per_minute: int):
        self.min_interval = 60.0 / per_minute
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Sleep just long enough to keep the minimum gap since the previous request."""
        with self._lock:
            delay = self._last + self.min_interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


class TokenRateLimiter:
    """Keeps the tokens sent in any 60-second window under `per_minute`, waiting when needed.

    Groq's free tier rejects calls once a minute's tokens pass its limit (8K), and a single
    Q&A call can use ~4K, so two quick questions are enough to hit it. This class keeps a
    list of (time, tokens) for recent calls. Before a call it reserves an estimate; once the
    real count is known, `settle()` replaces the estimate. The count lives in memory only, so
    it covers one process.
    """

    WINDOW = 60.0  # seconds

    def __init__(self, per_minute: int):
        self.limit = per_minute
        self._events: deque[list] = deque()  # [time, tokens], oldest first
        self._lock = threading.Lock()

    def acquire(self, estimated_tokens: int) -> list:
        """Wait until `estimated_tokens` fits in the window, reserve it, and return the reservation.

        Raises `ValueError` if one call alone is larger than the limit, since waiting would
        never help.
        """
        if estimated_tokens > self.limit:
            raise ValueError(
                f"one call of ~{estimated_tokens:,} tokens exceeds the {self.limit:,} tokens/min limit"
            )
        while True:
            with self._lock:
                now = time.monotonic()
                while self._events and self._events[0][0] <= now - self.WINDOW:
                    self._events.popleft()
                used = sum(tokens for _, tokens in self._events)
                if used + estimated_tokens <= self.limit:
                    entry = [now, estimated_tokens]
                    self._events.append(entry)
                    return entry
                # Find how long until enough of the oldest calls leave the window.
                need, freed, delay = used + estimated_tokens - self.limit, 0, 0.0
                for t, tokens in self._events:
                    freed += tokens
                    if freed >= need:
                        delay = t + self.WINDOW - now
                        break
            time.sleep(max(delay, 0.0) + 0.05)

    def settle(self, entry: list, actual_tokens: int) -> None:
        """Replace a reservation's estimate with the real token count of the finished call."""
        with self._lock:
            entry[1] = actual_tokens


def _estimate_tokens(text: str) -> int:
    """Roughly estimate the tokens in a text, to check the budget before a call is made."""
    return len(text) // 4  # about 4 characters per token in English


# --------------------------------------------------------------------------------------
# The interface
# --------------------------------------------------------------------------------------
class LLMProvider(ABC):
    """The common interface of all LLM backends: one `complete()` call.

    The rest of the project only talks to this class, so the vendor can be swapped without
    changing other code. The guards (budget, pacing, tokens per minute) live here so every
    backend gets them. `tokens_per_minute=None` means the backend has no per-minute token limit.
    """

    backend: str
    budget_class: type[DailyTokenBudget] = DailyTokenBudget  # how the daily budget is counted

    def __init__(self, model: str, daily_token_limit: int, requests_per_minute: int,
                 tokens_per_minute: int | None = None, account: str | None = None):
        """Set up the guards. `account` is a fingerprint of the API key, so the daily budget is per account."""
        self.model = model
        self.budget = self.budget_class(self.backend, daily_token_limit, account)
        self.pacer = RequestPacer(requests_per_minute)
        self.tpm = TokenRateLimiter(tokens_per_minute) if tokens_per_minute else None

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 1024) -> LLMResponse:
        """Send one prompt and return the answer.

        Checks the daily budget, waits for room in the tokens-per-minute window and for the
        pacer, calls the backend, then records the tokens used. `max_tokens` caps the length
        of the answer. If the call fails, its reservation keeps the estimate, because a failed
        call may still have counted against the provider's limit.
        """
        estimate = _estimate_tokens(prompt + (system or "")) + max_tokens
        self.budget.check(estimate)
        reservation = self.tpm.acquire(estimate) if self.tpm else None
        self.pacer.wait()
        resp = self._complete(prompt, system, max_tokens)
        if reservation is not None:
            self.tpm.settle(reservation, resp.total_tokens)
        self.budget.record(resp.total_tokens)
        return resp

    @abstractmethod
    def _complete(self, prompt: str, system: str | None, max_tokens: int) -> LLMResponse:
        """Backend-specific call: send the prompt to the actual LLM and return an `LLMResponse`."""


# --------------------------------------------------------------------------------------
# Backend A: Groq (the production path)
# --------------------------------------------------------------------------------------
class GroqProvider(LLMProvider):
    """Calls openai/gpt-oss-120b on Groq through its OpenAI-compatible API (free tier).

    Limits come from environment variables and default to the free tier's: 30 requests a
    minute, 8K tokens a minute and 200K tokens a day. Groq refills the daily tokens steadily
    (200K per 24 hours), so this backend uses a `TokenBucketBudget`.
    """

    backend = "groq"
    budget_class = TokenBucketBudget

    def __init__(self):
        """Read the settings from the environment and create the API client."""
        from openai import OpenAI

        super().__init__(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            daily_token_limit=int(os.getenv("GROQ_DAILY_TOKENS", "200000")),
            requests_per_minute=int(os.getenv("GROQ_RPM", "30")),
            tokens_per_minute=int(os.getenv("GROQ_TPM", "8000")),
            # Groq's limits are per account: count each API key separately (fingerprint only).
            account=hashlib.sha256(os.environ["GROQ_API_KEY"].encode()).hexdigest()[:8],
        )
        self.client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
            max_retries=5,  # the SDK retries rate-limit and server errors, waiting longer each time
            timeout=120,
        )
        # gpt-oss is a reasoning model: it "thinks" before answering, and those thinking
        # tokens count against the budget. "low" keeps that cost small.
        self.reasoning_effort = os.getenv("GROQ_REASONING_EFFORT", "low")

    def _complete(self, prompt, system, max_tokens):
        """Send the prompt to Groq and wrap the answer and token counts in an `LLMResponse`."""
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        r = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_tokens,  # this limit includes the thinking tokens
            reasoning_effort=self.reasoning_effort,
        )
        return LLMResponse(
            text=r.choices[0].message.content or "",
            input_tokens=r.usage.prompt_tokens,
            output_tokens=r.usage.completion_tokens,
            model=r.model,
            backend=self.backend,
        )


# --------------------------------------------------------------------------------------
# Backend B: Claude via `claude -p`  --  a WORKAROUND, not a production integration.
# It adds capacity on top of the free Groq tier (200K tokens/day), which is too small for the
# full corpus. A real deployment would call the Anthropic API with an API key instead.
# --------------------------------------------------------------------------------------
class ClaudeCLIProvider(LLMProvider):
    """WORKAROUND backend: runs Claude Code in headless mode (`claude -p`) on the Pro plan.

    It starts a `claude` process for each call and reads the answer and token usage from
    its JSON output. It is trimmed to the minimum (no tools, no project instructions) so
    each call costs few tokens.
    """

    backend = "claude_cli"

    def __init__(self):
        """Read the settings from the environment (model, daily cap, requests per minute)."""
        super().__init__(
            model=os.getenv("CLAUDE_MODEL", "sonnet"),  # Sonnet: extraction quality close to gpt-oss-120b
            # Pro-plan limits are per 5-hour window and are not published as token numbers,
            # so this daily cap is our own choice, to be tuned from real usage.
            daily_token_limit=int(os.getenv("CLAUDE_DAILY_TOKENS", "2000000")),
            requests_per_minute=int(os.getenv("CLAUDE_RPM", "10")),
        )

    def _complete(self, prompt, system, max_tokens):
        """Run `claude -p` with the prompt and wrap the answer and token counts in an `LLMResponse`.

        `claude -p` has no output-length option, so `max_tokens` only affects the budget check.
        """
        cmd = [
            "claude", "-p",
            "--model", self.model,
            "--output-format", "json",                         # includes the token usage
            "--system-prompt", system or "You are a helpful assistant.",  # replaces Claude Code's long default prompt
            "--tools", "",                                     # no tools, so no tool definitions are sent
            "--strict-mcp-config",                             # no MCP servers
            "--no-session-persistence",                        # do not save a session for each call
        ]
        # Run from an empty temporary folder so no project CLAUDE.md is loaded into the prompt.
        with tempfile.TemporaryDirectory() as cwd:
            proc = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True, cwd=cwd, timeout=300
            )
        if proc.returncode != 0:
            raise RuntimeError(f"claude -p failed ({proc.returncode}): {proc.stderr[-500:]}")
        out = json.loads(proc.stdout)
        if out.get("is_error"):
            raise RuntimeError(f"claude -p returned an error: {out.get('result')}")
        u = out.get("usage", {})
        input_tokens = (
            u.get("input_tokens", 0)
            + u.get("cache_creation_input_tokens", 0)
            + u.get("cache_read_input_tokens", 0)
        )
        model_used = ",".join(out.get("modelUsage", {}).keys()) or self.model
        return LLMResponse(
            text=out.get("result", ""),
            input_tokens=input_tokens,
            output_tokens=u.get("output_tokens", 0),
            model=model_used,
            backend=self.backend,
        )


# --------------------------------------------------------------------------------------
# Backend C: Google Antigravity via `agy`  --  also a WORKAROUND, not a production integration.
# Its 5-hour and weekly quotas are separate from Claude's, so it adds capacity on top of
# Groq and Claude. A real deployment would call the Gemini API with an API key instead.
# --------------------------------------------------------------------------------------
# A custom `agy` agent with no tools and none of the default components. Without it every call
# carries about 11K tokens of Antigravity's own agent prompt and tool definitions. The system
# prompt of each call is appended below the front matter.
_AGY_AGENT_NAME = "plain"
_AGY_AGENT_FILE = """---
name: {name}
description: Plain text-in, text-out assistant with no tools
mainAgent: true
tools: []
excludeDefaultComponents: true
---
{system}
"""


class AntigravityCLIProvider(LLMProvider):
    """WORKAROUND backend: runs Google Antigravity in headless mode (`agy`) with a Gemini model.

    It starts an `agy` process for each call and reads the answer and token usage from its JSON
    output. Each call runs in an empty temporary folder that holds a minimal agent definition
    (no tools, no default prompt), so a call costs a few hundred tokens instead of ~11K. The
    prompt is sent on stdin, which has no practical size limit.
    """

    backend = "antigravity_cli"

    def __init__(self):
        """Read the settings from the environment (model, daily cap, requests per minute)."""
        super().__init__(
            model=os.getenv("AGY_MODEL", "gemini-4-hydrogen-medium"),
            # The 5-hour and weekly quotas are not published as token numbers, so this daily
            # cap is our own choice, to be tuned from real usage (`agy -p /usage` shows the meters).
            daily_token_limit=int(os.getenv("AGY_DAILY_TOKENS", "2000000")),
            requests_per_minute=int(os.getenv("AGY_RPM", "10")),
        )

    def _complete(self, prompt, system, max_tokens):
        """Run `agy` with the prompt and wrap the answer and token counts in an `LLMResponse`.

        `agy` has no output-length option, so `max_tokens` only affects the budget check.
        """
        cmd = [
            "agy",
            "--agent", _AGY_AGENT_NAME,
            "--model", self.model,
            "--output-format", "json",  # includes the token usage
        ]
        # The agent file lives in the temporary folder, which is also the working folder, so no
        # global Antigravity setup is needed and no project files are picked up.
        with tempfile.TemporaryDirectory() as cwd:
            agents_dir = Path(cwd) / ".agents" / "agents"
            agents_dir.mkdir(parents=True)
            (agents_dir / f"{_AGY_AGENT_NAME}.md").write_text(
                _AGY_AGENT_FILE.format(name=_AGY_AGENT_NAME, system=system or "You are a helpful assistant.")
            )
            proc = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True, cwd=cwd, timeout=300
            )
        if proc.returncode != 0:
            raise RuntimeError(f"agy failed ({proc.returncode}): {(proc.stderr or proc.stdout)[-500:]}")
        out = json.loads(proc.stdout.strip().splitlines()[0])
        if out.get("status") != "SUCCESS":
            raise RuntimeError(f"agy returned status {out.get('status')!r}: {str(out)[:500]}")
        u = out.get("usage", {})
        return LLMResponse(
            text=out.get("response", ""),
            input_tokens=u.get("input_tokens", 0) + u.get("cache_read_tokens", 0),
            output_tokens=u.get("output_tokens", 0),  # already includes the thinking tokens
            model=self.model,
            backend=self.backend,
        )


# --------------------------------------------------------------------------------------
# Factory: picks the backend
# --------------------------------------------------------------------------------------
# The available backends by name, for `get_llm()` and the smoke test.
_BACKENDS = {
    "groq": GroqProvider,
    "claude_cli": ClaudeCLIProvider,
    "antigravity_cli": AntigravityCLIProvider,
}


def get_llm(backend: str | None = None) -> LLMProvider:
    """Return the backend named by `backend` or, if not given, by the LLM_BACKEND environment variable (default "groq").

    Other modules call this instead of creating a provider themselves, so the backend can be
    switched in `.env` without touching code.
    """
    name = backend or os.getenv("LLM_BACKEND", "groq")
    if name not in _BACKENDS:
        raise ValueError(f"Unknown LLM_BACKEND {name!r}; choose from {list(_BACKENDS)}")
    return _BACKENDS[name]()


# --------------------------------------------------------------------------------------
# Smoke test: sends the same prompt to both backends.  Run: .venv/bin/python -m sec10k.llm.providers
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    system = "You answer in one short sentence."
    prompt = "Name one main competitor of NVIDIA in data-center GPUs."
    for name in _BACKENDS:
        llm = get_llm(name)
        t = time.time()
        try:
            r = llm.complete(prompt, system=system, max_tokens=512)
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}\n")
            continue
        print(f"[{r.backend}] model={r.model}  {time.time() - t:.1f}s")
        print(f"  answer: {r.text.strip()}")
        print(f"  tokens: in={r.input_tokens:,} out={r.output_tokens:,}  "
              f"(used today: {llm.budget.used_today():,} / {llm.budget.daily_limit:,})\n")
