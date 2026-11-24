"""LLM provider interface: one `complete()` call, swappable backends.

The rest of the codebase only talks to `LLMProvider`. Which backend answers is chosen by
the LLM_BACKEND environment variable (see `get_llm()`), so switching vendors needs no code change.

Backends:
  - GroqProvider       the "real" production path (openai/gpt-oss-120b, OpenAI-compatible API)
  - ClaudeCLIProvider  WORKAROUND for free-API limits: shells out to `claude -p` (Claude Pro plan)

Guards (minimal Phase 0 version; Phase 4 replaces these with a proper limiter, a token-per-minute
budget and a response cache):
  - retries with backoff   -> openai SDK's built-in `max_retries` (Groq only)
  - request pacing         -> at most N requests/minute, per backend
  - daily token budget     -> refuses to call once today's tokens would exceed the limit;
                              persisted to a JSON file so it survives restarts
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

from sec10k.config import STATE_DIR  # also loads .env


# --------------------------------------------------------------------------------------
# Result type: every backend returns the same shape
# --------------------------------------------------------------------------------------
@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    backend: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BudgetExceeded(RuntimeError):
    """Raised instead of calling the LLM when today's token budget is used up."""


# --------------------------------------------------------------------------------------
# Minimal guards (Phase 0). Phase 4 replaces these.
# --------------------------------------------------------------------------------------
class DailyTokenBudget:
    """Counts tokens used per backend per day in a small JSON file."""

    def __init__(self, backend: str, daily_limit: int):
        self.backend = backend
        self.daily_limit = daily_limit
        self.path = STATE_DIR / f"budget_{backend}.json"

    def _load(self) -> dict:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            if data.get("date") == date.today().isoformat():
                return data
        return {"date": date.today().isoformat(), "tokens_used": 0, "requests": 0}

    def used_today(self) -> int:
        return self._load()["tokens_used"]

    def check(self, estimated_tokens: int) -> None:
        used = self.used_today()
        if used + estimated_tokens > self.daily_limit:
            raise BudgetExceeded(
                f"{self.backend}: {used:,} tokens used today + ~{estimated_tokens:,} estimated "
                f"> daily limit {self.daily_limit:,}"
            )

    def record(self, tokens: int) -> None:
        data = self._load()
        data["tokens_used"] += tokens
        data["requests"] += 1
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(self.path)  # atomic: a crash never leaves a half-written file


class RequestPacer:
    """Spaces requests so we never exceed `per_minute` requests per minute."""

    def __init__(self, per_minute: int):
        self.min_interval = 60.0 / per_minute
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            delay = self._last + self.min_interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


def _estimate_tokens(text: str) -> int:
    return len(text) // 4  # rough rule of thumb: ~4 characters per token in English


# --------------------------------------------------------------------------------------
# The interface
# --------------------------------------------------------------------------------------
class LLMProvider(ABC):
    backend: str

    def __init__(self, model: str, daily_token_limit: int, requests_per_minute: int):
        self.model = model
        self.budget = DailyTokenBudget(self.backend, daily_token_limit)
        self.pacer = RequestPacer(requests_per_minute)

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 1024) -> LLMResponse:
        """Send one prompt, get one answer. Guards run here, so every backend gets them."""
        estimate = _estimate_tokens(prompt + (system or "")) + max_tokens
        self.budget.check(estimate)
        self.pacer.wait()
        resp = self._complete(prompt, system, max_tokens)
        self.budget.record(resp.total_tokens)
        return resp

    @abstractmethod
    def _complete(self, prompt: str, system: str | None, max_tokens: int) -> LLMResponse: ...


# --------------------------------------------------------------------------------------
# Backend A: Groq (production path)
# --------------------------------------------------------------------------------------
class GroqProvider(LLMProvider):
    backend = "groq"

    def __init__(self):
        from openai import OpenAI

        super().__init__(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            daily_token_limit=int(os.getenv("GROQ_DAILY_TOKENS", "200000")),
            requests_per_minute=int(os.getenv("GROQ_RPM", "30")),
        )
        self.client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
            max_retries=5,  # SDK retries 429/5xx with exponential backoff (honours Retry-After)
            timeout=120,
        )
        # gpt-oss is a reasoning model: it "thinks" before answering, and those thinking tokens
        # count against our budget. "low" keeps that overhead small.
        self.reasoning_effort = os.getenv("GROQ_REASONING_EFFORT", "low")

    def _complete(self, prompt, system, max_tokens):
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        r = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_tokens,  # includes reasoning tokens
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
# Backend B: Claude via `claude -p`  —  WORKAROUND, not a production integration.
# We use it because the free Groq tier (200K tokens/day) is too small for the full corpus.
# A real deployment would call the Anthropic API with an API key instead.
# --------------------------------------------------------------------------------------
class ClaudeCLIProvider(LLMProvider):
    backend = "claude_cli"

    def __init__(self):
        super().__init__(
            model=os.getenv("CLAUDE_MODEL", "haiku"),
            # Pro-plan limits are per 5-hour window and not published as token numbers;
            # this is our own self-imposed daily cap, tune it once we see real usage.
            daily_token_limit=int(os.getenv("CLAUDE_DAILY_TOKENS", "2000000")),
            requests_per_minute=int(os.getenv("CLAUDE_RPM", "10")),
        )

    def _complete(self, prompt, system, max_tokens):
        # max_tokens: `claude -p` has no output-length flag, so it is only used for budgeting.
        cmd = [
            "claude", "-p",
            "--model", self.model,
            "--output-format", "json",                         # gives us token usage
            "--system-prompt", system or "You are a helpful assistant.",  # replace CC's big prompt
            "--tools", "",                                     # no tools -> no tool definitions sent
            "--strict-mcp-config",                             # no MCP servers
            "--no-session-persistence",                        # don't save a session per call
        ]
        # Run from an empty temp dir so no project CLAUDE.md gets loaded into the context.
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
# Factory
# --------------------------------------------------------------------------------------
_BACKENDS = {"groq": GroqProvider, "claude_cli": ClaudeCLIProvider}


def get_llm(backend: str | None = None) -> LLMProvider:
    """Return the configured backend (LLM_BACKEND env var, default 'groq')."""
    name = backend or os.getenv("LLM_BACKEND", "groq")
    if name not in _BACKENDS:
        raise ValueError(f"Unknown LLM_BACKEND {name!r}; choose from {list(_BACKENDS)}")
    return _BACKENDS[name]()


# --------------------------------------------------------------------------------------
# Smoke test: same prompt to both backends.  Run: .venv/bin/python -m sec10k.llm.providers
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
