"""Stage 3 of the extraction pipeline: group ready work items into LLM calls.

What it does: for one backend, claims the next batch of ready items from the job store and
builds the prompt for it. A batch holds items of **one company and one route** and may span
several of that company's fiscal years: route A takes up to 50 risk titles per call, route B
up to 8 chunks per call (25 and 4 on Groq, whose 8K tokens-per-minute limit caps one call).
Filling batches across years matters because, after a company's first year, most filings
have only a few new risk titles; one call per filing would mostly pay for the system prompt.

A batch never mixes companies, because "SELF" in the LLM's answer stands for the filing
company. Each backend works on its own companies (assigned by the router), so the batcher
takes the list of companies it may draw from.

Why it is needed: it sits between the job store (what is ready) and the worker (which makes
the call), and decides how much text goes into one request, which sets the cost per item.
"""

from __future__ import annotations

from dataclasses import dataclass

from sec10k.extract.llm_extract import (BATCH_BY_BACKEND, BATCH_DEFAULT, SYSTEM_A, SYSTEM_B,
                                        filer_name, prompt_a, prompt_b)
from sec10k.extraction_pipeline.jobs import JobStore
from sec10k.ingest.chunker import Chunk, iter_chunks


@dataclass
class Batch:
    """One LLM call's worth of work: the claimed items, their texts and the finished prompt.

    `chunks` holds, for route B, the chunk of each item in the same order as the prompt's
    passages (the worker validates each answer against its chunk). For route A it is empty
    and the titles are in the items. `max_tokens` is the output limit for the call.
    """

    route: str
    ticker: str
    backend: str
    items: list
    chunks: list[Chunk]
    system: str
    prompt: str
    max_tokens: int


class Batcher:
    """Builds batches for one backend from the job store.

    Keeps the chunks of the company it is working on in memory, because route-B prompts need
    the chunk text and the job store only holds chunk IDs. A company's chunks are loaded
    once (a few seconds) and kept until the batcher moves to another company.
    """

    def __init__(self, store: JobStore, backend: str):
        """Create a batcher that claims items for `backend` and sizes batches for it."""
        self.store = store
        self.backend = backend
        self.titles_per_call, self.chunks_per_call, self.max_tokens = BATCH_BY_BACKEND.get(backend, BATCH_DEFAULT)
        self._ticker: str | None = None
        self._chunks: dict[str, Chunk] = {}

    def _company_chunks(self, ticker: str) -> dict[str, Chunk]:
        """The company's chunks by chunk_id, loaded once per company."""
        if ticker != self._ticker:
            self._chunks = {c.chunk_id: c for c in iter_chunks([ticker])}
            self._ticker = ticker
        return self._chunks

    def next_batch(self, route: str, tickers: list[str]) -> Batch | None:
        """Claim and build the next batch of `route` from the given companies, or None if none is ready.

        Takes the first company (in the given order) that has ready items, so a backend
        finishes one company before it starts the next. The claimed items are "running" in
        the job store from here on; the worker marks them done or failed.
        """
        ready = set(self.store.ready_tickers(route))
        ticker = next((t for t in tickers if t in ready), None)
        if ticker is None:
            return None
        size = self.titles_per_call if route == "A" else self.chunks_per_call
        name = filer_name(ticker)
        # Load the chunk texts before claiming, so a failure here leaves no item stuck "running".
        by_id = self._company_chunks(ticker) if route == "B" else {}
        items = self.store.claim(route, ticker, size, self.backend)
        if not items:                       # another worker took them in between
            return None
        if route == "A":
            return Batch("A", ticker, self.backend, items, [], SYSTEM_A,
                         prompt_a(name, ticker, [r["title"] for r in items]), self.max_tokens)
        missing = [r["item_id"] for r in items if r["chunk_id"] not in by_id]
        if missing:                         # chunking changed since planning: these items can't be built
            self.store.mark_failed(missing, "chunk not found (chunking changed since planning)")
            items = [r for r in items if r["chunk_id"] in by_id]
            if not items:
                return None
        chunks = [by_id[r["chunk_id"]] for r in items]
        return Batch("B", ticker, self.backend, items, chunks, SYSTEM_B,
                     prompt_b(name, ticker, chunks), self.max_tokens)
