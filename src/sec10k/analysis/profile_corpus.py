"""Phase 1: profile the 10-K corpus before designing anything.

Step 1.1: section quality checks.
Step 1.2: real token counts (gpt-oss tokenizer), by section, year and company.
Step 1.3: year-over-year repetition (how much text is copied from the previous filing).
Step 1.4: cost estimate (days to finish per LLM backend).
Run from the project root:  .venv/bin/python -m sec10k.analysis.profile_corpus

Reads only the filings listed in manifest.csv (the authoritative list) and writes
detailed results to data/profile/ (git-ignored). The corpus itself is never modified.
"""

import json
import re

import pandas as pd
import tiktoken

from sec10k.config import CORPUS, SECTIONS, load_manifest
from sec10k.config import PROFILE_DIR as OUT
from sec10k.ingest.loader import NUMLINE_RE, PAGE_RE

# gpt-oss-120b's own tokenizer, so counts match what Groq bills against the daily budget.
# Claude uses a different tokenizer (not available locally); we measure that ratio later.
ENC = tiktoken.get_encoding("o200k_harmony")

# Expected opening heading per section, e.g. "Item 1A. Risk Factors" / "ITEM 1A -".
HEADING = {
    "item_1": r"item\s*1\b(?!\s*a)",
    "item_1A": r"item\s*1a\b",
    "item_7": r"item\s*7\b(?!\s*a)",
    "item_7A": r"item\s*7a\b",
}
# Real HTML leftovers only. A generic "<...>" pattern also matches product text such as
# "<Dial>, <Record>" (seen in Twilio filings), so we list actual tag names.
HTML_RE = re.compile(
    r"</?(?:div|span|p|br|td|tr|th|table|font|b|i|u|a|img|sup|sub|hr)\b[^>]*>|&(?:nbsp|amp|lt|gt|#\d+);",
    re.I,
)
# A later item's heading inside the section suggests the extractor ran past the end.
NEXT_ITEM_RE = re.compile(r"(?im)^\s*item\s*(?:8|2)\s*[.:\-—–]")


def iter_sections(manifest: pd.DataFrame):
    """Yield (manifest_row, section_name, text) for every section of every filing."""
    for row in manifest.itertuples(index=False):
        filing = json.loads((CORPUS / row.json_path).read_text())
        for sec in SECTIONS:
            yield row, sec, filing.get(sec) or ""


def section_metrics(sec: str, text: str) -> dict:
    lines = [ln for ln in text.split("\n") if ln.strip()]
    n_lines = len(lines) or 1
    return {
        "words": len(text.split()),
        "tokens": len(ENC.encode_ordinary(text)),
        "starts_with_heading": bool(re.match(r"\s*" + HEADING[sec], text, re.I)),
        "html_hits": len(HTML_RE.findall(text)),
        "page_artifact_lines": sum(bool(PAGE_RE.match(ln)) for ln in lines),
        "numeric_line_share": round(sum(bool(NUMLINE_RE.match(ln.strip())) for ln in lines) / n_lines, 3),
        "next_item_heading": bool(NEXT_ITEM_RE.search(text)),
        "ends_with_punct": text.rstrip()[-1:] in {".", ":", "”", '"', ")"},
    }


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    # Compare each section with the same company's other years: 10-K sections are fairly
    # stable per company, so a big jump either way usually means an extraction error.
    med = df.groupby(["cik", "section"])["words"].transform("median")
    df["ratio_to_company_median"] = (df["words"] / med).round(2)

    flags = {
        "empty_or_heading_only": df["words"] < 50,
        "much_shorter_than_usual": (df["ratio_to_company_median"] < 0.33) & (df["words"] >= 50),
        "much_longer_than_usual": df["ratio_to_company_median"] > 3,
        "missing_heading": ~df["starts_with_heading"],
        "html_leftovers": df["html_hits"] > 0,
        "next_item_inside": df["next_item_heading"],
        "flattened_tables": df["numeric_line_share"] > 0.2,
        "page_artifacts": df["page_artifact_lines"] > 0,
    }
    for name, mask in flags.items():
        df[f"flag_{name}"] = mask
    return df


def report_quality(df: pd.DataFrame) -> None:
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    print("Words per section:")
    print(df.groupby("section")["words"].describe(percentiles=[0.05, 0.5, 0.95]).round(0).astype(int), "\n")
    print("Flagged sections (count per section type):")
    summary = df.groupby("section")[flag_cols].sum().T
    summary.index = summary.index.str.removeprefix("flag_")
    print(summary, "\n")

    serious = ["flag_empty_or_heading_only", "flag_much_shorter_than_usual",
               "flag_much_longer_than_usual", "flag_next_item_inside"]
    bad = df[df[serious].any(axis=1)]
    print(f"Likely extraction errors ({len(bad)} sections):")
    cols = ["ticker", "fiscal_year", "section", "words", "ratio_to_company_median"] + serious
    print(bad[cols].rename(columns=lambda c: c.removeprefix("flag_")).to_string(index=False), "\n")


def report_tokens(df: pd.DataFrame, manifest: pd.DataFrame) -> None:
    m = 1e6
    total = df["tokens"].sum()
    est = manifest["est_tokens_all"].sum()
    print("=" * 70, "\nTOKENS (o200k_harmony = gpt-oss tokenizer)\n")
    print(f"Total: {total / m:.1f}M tokens  (manifest estimate: {est / m:.1f}M, "
          f"real/estimate = {total / est:.2f})")
    print(f"Tokens per word: {total / df['words'].sum():.2f}\n")

    by_sec = df.groupby("section")["tokens"].agg(["sum", "median", "max"])
    by_sec["share"] = (by_sec["sum"] / total).round(3)
    by_sec["sum"] = (by_sec["sum"] / m).round(2)
    print("By section (sum in M; median/max per filing):")
    print(by_sec.rename(columns={"sum": "sum_M"}), "\n")

    per_filing = df.groupby("json_path")["tokens"].sum()
    print("Per filing:", per_filing.describe(percentiles=[0.05, 0.5, 0.95]).round(0).astype(int).to_dict(), "\n")

    by_year = df.pivot_table(index="fiscal_year", columns="section", values="tokens", aggfunc="sum") / m
    by_year["total_M"] = by_year.sum(axis=1)
    by_year["filings"] = df.groupby("fiscal_year")["json_path"].nunique()
    print("By fiscal year (M tokens):")
    print(by_year.round(2), "\n")

    by_co = df.groupby("ticker").agg(tokens=("tokens", "sum"), filings=("json_path", "nunique"))
    by_co["per_filing"] = (by_co["tokens"] / by_co["filings"]).round(0).astype(int)
    print("By company: per-filing tokens", by_co["per_filing"].describe().round(0).astype(int).to_dict())
    print("Largest per filing:\n", by_co.nlargest(5, "per_filing").to_string())
    print("Smallest per filing:\n", by_co.nsmallest(5, "per_filing").to_string(), "\n")

    OUT.mkdir(parents=True, exist_ok=True)
    by_co.sort_values("tokens", ascending=False).to_csv(OUT / "tokens_by_company.csv")
    by_year.to_csv(OUT / "tokens_by_year.csv")


def paragraphs(text: str) -> list[str]:
    """Split a section into paragraphs (one per line), dropping page furniture and
    number-only table lines: both are noise and would hide real repetition."""
    return [ln.strip() for ln in text.split("\n")
            if ln.strip() and not PAGE_RE.match(ln) and not NUMLINE_RE.match(ln.strip())]


def norm(par: str, mask_numbers: bool) -> str:
    """Normalize so cosmetic differences don't count as changes."""
    p = par.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    p = re.sub(r"[–—]", "-", p)
    p = re.sub(r"\s+", " ", p)
    if mask_numbers:  # "fiscal 2023" == "fiscal 2024"; "$5.1 billion" == "$6.3 billion"
        p = re.sub(r"\d[\d,.]*", "#", p)
    return p


# Sentence boundary: end punctuation, whitespace, then something that starts a sentence.
SENT_SPLIT_RE = re.compile(r"(?<=[.;!?])\s+(?=[A-Z\"“(•])")
# Two match units x two normalizations. Paragraph matching shows what we'd skip if we
# compare whole paragraphs; sentence matching shows how much text is *really* reused
# (a paragraph with one edited sentence counts as fully new at the paragraph level).
VARIANTS = {"para_exact": ("para", False), "para_masked": ("para", True),
            "sent_exact": ("sent", False), "sent_masked": ("sent", True)}


def units(text: str) -> dict[str, list[str]]:
    pars = paragraphs(text)
    return {"para": pars, "sent": [x for p in pars for x in SENT_SPLIT_RE.split(p) if x.strip()]}


def repetition_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """For each filing-section: tokens in units (paragraphs or sentences) already present
    (same company, same section) in that company's previous filing in the manifest."""
    rows = []
    prev: dict[tuple, dict[str, set]] = {}  # (cik, section) -> {variant: set of normalized units}
    ordered = manifest.sort_values(["cik", "fiscal_year"])
    for row, sec, text in iter_sections(ordered):
        u = units(text)
        toks = {kind: [len(ENC.encode_ordinary(x)) for x in xs] for kind, xs in u.items()}
        seen = prev.get((row.cik, sec))
        rec = {"cik": row.cik, "ticker": row.ticker, "fiscal_year": row.fiscal_year,
               "section": sec, "has_prev": seen is not None,
               "paragraphs": len(u["para"]), "tokens": sum(toks["para"])}
        current = {}
        for var, (kind, mask) in VARIANTS.items():
            keys = [norm(x, mask) for x in u[kind]]
            current[var] = set(keys)
            rec[f"rep_{var}"] = sum(t for k, t in zip(keys, toks[kind]) if k in seen[var]) if seen else 0
        prev[(row.cik, sec)] = current
        rows.append(rec)
    return pd.DataFrame(rows)


def report_repetition(rep: pd.DataFrame) -> None:
    m = 1e6
    rep_cols = [f"rep_{v}" for v in VARIANTS]
    print("=" * 70, "\nYEAR-OVER-YEAR REPETITION (same company + section vs previous filing)\n")
    total = rep["tokens"].sum()
    print(f"Tokens after dropping page/table noise: {total / m:.1f}M")
    later = rep[rep["has_prev"]]
    print(f"Filings with a previous year: {later.groupby(['cik', 'fiscal_year']).ngroups} "
          f"of {rep.groupby(['cik', 'fiscal_year']).ngroups}\n")

    def shares(g: pd.DataFrame) -> pd.DataFrame:
        return g[rep_cols].div(g["tokens"], axis=0).round(3).rename(columns=lambda c: c[4:])

    by_sec = later.groupby("section")[["tokens"] + rep_cols].sum()
    by_sec.loc["ALL"] = by_sec.sum()
    print("Share of tokens repeated from previous filing (filings that have one):")
    print(shares(by_sec).assign(tokens_M=(by_sec["tokens"] / m).round(2)), "\n")

    by_year = later.groupby("fiscal_year")[["tokens"] + rep_cols].sum()
    print("By fiscal year (share repeated):")
    print(shares(by_year).T.to_string(), "\n")

    # Spread across companies: is repetition typical or driven by a few heavy copiers?
    by_co = later.groupby("ticker")[["tokens"] + rep_cols].sum()
    print("Per company (spread of share repeated):")
    print(shares(by_co).describe(percentiles=[0.1, 0.5, 0.9]).round(2).loc[["min", "10%", "50%", "90%", "max"]], "\n")

    # What the LLM would have to read: first filings in full + only the new units after.
    print("Tokens the LLM must read once (first filing in full + only new text after):")
    for var in VARIANTS:
        need = total - rep[f"rep_{var}"].sum()
        print(f"  {var:12s}: {need / m:5.1f}M  ({need / total:.0%} of corpus)"
              f"  ≈ {need / 200_000:.0f} days at Groq's 200K tokens/day (input only)")

    OUT.mkdir(parents=True, exist_ok=True)
    rep.to_csv(OUT / "repetition_by_section.csv", index=False)


# ---- Step 1.4 cost model ------------------------------------------------------------
# MEASURED (2026-11-24, one 3K-token sample of 10-K text, sec10k/llm/providers.py):
GROQ_TOKEN_RATIO = 1.00      # Groq prompt_tokens / o200k_harmony tokens
GROQ_CALL_OVERHEAD = 84      # chat-format tokens per call
CLAUDE_TOKEN_RATIO = 1.62    # Claude Haiku input tokens / o200k_harmony tokens (single sample!)
CLAUDE_CALL_OVERHEAD = 621   # fixed `claude -p` input tokens per call (with our minimal flags)
# ASSUMED (to be replaced with real numbers in Phase 3):
CONTENT_PER_CALL = 4_000     # o200k tokens of filing text per request (fits Groq's 8K tokens/min)
PROMPT_TOKENS = 1_000        # extraction instructions + schema per request
OUTPUT_SHARE = 0.25          # output tokens as a share of content tokens (JSON triples)
GROQ_REASONING = 300         # gpt-oss "thinking" tokens per call at reasoning_effort=low
# BUDGETS (tokens/day): Groq free tier; Claude = our self-imposed cap (Pro limits are not
# published as token numbers, so the real figure must be measured in Phase 4).
GROQ_DAILY = 200_000
CLAUDE_DAILY = 2_000_000


def content_per_day() -> dict[str, float]:
    """How many o200k tokens of *filing text* each backend gets through per day, once
    prompt, output and per-call overhead are paid for."""
    groq_call = (GROQ_CALL_OVERHEAD + (PROMPT_TOKENS + CONTENT_PER_CALL) * GROQ_TOKEN_RATIO
                 + OUTPUT_SHARE * CONTENT_PER_CALL * GROQ_TOKEN_RATIO + GROQ_REASONING)
    claude_call = (CLAUDE_CALL_OVERHEAD + (PROMPT_TOKENS + CONTENT_PER_CALL) * CLAUDE_TOKEN_RATIO
                   + OUTPUT_SHARE * CONTENT_PER_CALL * CLAUDE_TOKEN_RATIO)
    return {
        "groq": GROQ_DAILY / groq_call * CONTENT_PER_CALL,
        "claude": CLAUDE_DAILY / claude_call * CONTENT_PER_CALL,
        "_groq_call": groq_call, "_claude_call": claude_call,
    }


def report_cost_estimate(rep: pd.DataFrame) -> None:
    m = 1e6
    total = rep["tokens"].sum()
    to_read = {  # o200k tokens of filing text the LLM must see once
        "no skipping": total,
        "skip unchanged paragraphs": total - rep["rep_para_exact"].sum(),
        "skip unchanged sentences": total - rep["rep_sent_exact"].sum(),
    }
    speed = content_per_day()
    both = speed["groq"] + speed["claude"]
    print("=" * 70, "\nCOST ESTIMATE (input + output + overhead, in days to finish)\n")
    print(f"Per request of {CONTENT_PER_CALL:,} content tokens: Groq {speed['_groq_call']:,.0f} tokens, "
          f"Claude {speed['_claude_call']:,.0f} tokens (Claude tokenizer x{CLAUDE_TOKEN_RATIO})")
    print(f"Filing text per day: Groq {speed['groq'] / 1e3:,.0f}K, Claude {speed['claude'] / 1e3:,.0f}K "
          f"(at {CLAUDE_DAILY / m:.0f}M/day cap), both {both / 1e3:,.0f}K\n")

    rows = []
    for name, tokens in to_read.items():
        for keep in (1.0, 0.5, 0.3):  # share of chunks that pass the Phase 3 tier-2/3 filter
            t = tokens * keep
            rows.append({"skipping": name, "filter_keeps": f"{keep:.0%}", "text_M": round(t / m, 1),
                         "groq_days": round(t / speed["groq"]), "claude_days": round(t / speed["claude"]),
                         "both_days": round(t / both)})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nfilter_keeps = share of chunks the Phase 3 pattern/NLP filter sends to the LLM (unknown yet)")


def main() -> None:
    manifest = load_manifest()
    rows = []
    for row, sec, text in iter_sections(manifest):
        rows.append({
            "cik": row.cik, "ticker": row.ticker, "fiscal_year": row.fiscal_year,
            "json_path": row.json_path, "section": sec, **section_metrics(sec, text),
        })
    df = add_flags(pd.DataFrame(rows))

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "section_quality.csv", index=False)

    print(f"Filings: {len(manifest)}  |  sections checked: {len(df)}\n")
    report_quality(df)
    report_tokens(df, manifest)
    rep = repetition_table(manifest)
    report_repetition(rep)
    report_cost_estimate(rep)
    print(f"Details: {OUT}/ (section_quality.csv, tokens_by_company.csv, tokens_by_year.csv, repetition_by_section.csv)")


if __name__ == "__main__":
    main()
