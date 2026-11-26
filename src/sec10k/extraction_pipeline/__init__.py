"""The production LLM extraction pipeline: runs the tier-4 extraction over the whole corpus.

The stages, in the order data flows through them:
planner (what needs extracting) -> job store (what is pending, done or failed) -> batcher
(groups items into LLM calls) -> router (picks a backend that has budget left) -> worker
(calls, validates, retries) -> progress reporting. The graph writer that loads the results
into Neo4j lives in `sec10k.graph`.

Why it is needed: extracting all 950 filings takes weeks of free-tier quota, so the run must
be resumable, budgeted and never pay twice for the same text. `sec10k.extract` holds *how*
one batch is extracted (prompts, validation); this package runs that at scale.
"""
