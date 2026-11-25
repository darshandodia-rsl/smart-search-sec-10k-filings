"""Plain-RAG question answering: vector search, then an LLM answer that cites its sources.

What it does: finds the passages most similar to a question (`vector_search.search`),
numbers them [1]..[k] with their source (company, fiscal years, section, heading), and asks
the LLM to answer using only those passages, marking every claim with the number of the
passage it came from.

Why it is needed: this is the complete plain-RAG baseline of Phase 2. The graph-based
retrieval of Phase 6 is judged by whether it answers the same questions better than this.

Size of one call: 8 passages of up to ~400 tokens plus instructions is ~3K input tokens, and
the answer (including the model's thinking) is capped at `MAX_ANSWER_TOKENS`. That keeps a
question under Groq's free-tier limit of 8K tokens per minute.

Run from the project root:  .venv/bin/python -m sec10k.retrieval.qa "question" [-k 8] [--backend groq]
"""

import argparse
from dataclasses import dataclass

from sec10k.llm.providers import LLMResponse, get_llm
from sec10k.retrieval.vector_search import Hit, search

# Upper limit on the answer length. For gpt-oss this includes its thinking tokens.
MAX_ANSWER_TOKENS = 1500

SYSTEM_PROMPT = """You answer questions about US tech companies using excerpts from their \
SEC 10-K filings.
Rules:
- Use ONLY the numbered sources given. Do not use outside knowledge.
- After every claim, cite the source number(s) in square brackets, e.g. [2] or [1][4].
- Mention the company and fiscal year(s) a fact comes from.
- If the sources do not answer the question, or only answer part of it, say so plainly.
- Be concise."""


@dataclass
class Answer:
    """The result of one question: the LLM's answer, the passages it was given, and the call details.

    `hits[i]` is source [i + 1] in the answer text, so a citation like [3] can be looked up
    as `hits[2]`.
    """
    question: str
    text: str
    hits: list[Hit]
    llm: LLMResponse | None


def build_prompt(question: str, hits: list[Hit]) -> str:
    """Write the user prompt: the numbered sources, each with its citation, then the question."""
    sources = "\n\n".join(
        f"[{i}] {h.company} ({h.citation()}) | {h.heading or '-'}\n{h.text}"
        for i, h in enumerate(hits, 1)
    )
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


def answer(question: str, k: int = 8, backend: str | None = None) -> Answer:
    """Answer `question` from the `k` best passages, using the given (or default) LLM backend.

    If the search finds nothing, the LLM is not called, so no tokens are spent.
    """
    hits = search(question, k)
    if not hits:
        return Answer(question, "No relevant passages found.", [], None)
    resp = get_llm(backend).complete(build_prompt(question, hits), system=SYSTEM_PROMPT,
                                     max_tokens=MAX_ANSWER_TOKENS)
    return Answer(question, resp.text.strip(), hits, resp)


def main() -> None:
    """Answer one question from the command line and print the answer, its sources and token use."""
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=8, help="number of passages given to the LLM")
    ap.add_argument("--backend", help="LLM backend (default: LLM_BACKEND from .env)")
    a = ap.parse_args()
    ans = answer(a.question, a.k, a.backend)
    print(f"Q: {ans.question}\n\n{ans.text}\n\nSources:")
    for i, h in enumerate(ans.hits, 1):
        print(f"  [{i}] {h.score:.3f}  {h.citation()}  | {h.heading or '-'}")
    if ans.llm:
        print(f"\nTokens: in={ans.llm.input_tokens:,} out={ans.llm.output_tokens:,}  "
              f"({ans.llm.backend}, {ans.llm.model})")


if __name__ == "__main__":
    main()
