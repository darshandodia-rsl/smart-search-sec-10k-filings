"""Step 2: pick the final 130 companies.

This step was originally done by hand; this script reproduces it exactly
(verified: same companies.csv, companies_all_ranked.csv and scripts/ciks.txt).

Inputs : scripts/step1_raw.csv  (step 1: software SIC 7370-7374 with an FY2024 10-K)
         scripts/extra.csv      (step 1b: hand-picked hardware/security names)
Outputs: scripts/companies_all_ranked.csv, companies.csv (top N), scripts/ciks.txt
"""
from _corpus_env import CORPUS_DIR  # loads .env, cd into corpus folder

import pandas as pd

TOP_N = 130

# Hand-picked additions outside the software SIC codes (from step 1b): PANW, FTNT (SIC 3577), AAPL (3571).
EXTRA_FILE = "scripts/extra.csv"

# XBRL data errors in the SEC "frames" API, found by eye:
# HubSpot reported its float in thousands-scaled units -> $29 trillion instead of $29 billion.
FLOAT_DIVIDE_BY_1000 = {1404655: "HUBS"}
# Implausible floats for the company's size, so excluded rather than guessed.
EXCLUDE = {
    1340476: "DRTTF  DIRTT Environmental Solutions ($33B float reported)",
    1843714: "ZPTA   Zapata Quantum ($6.7B float reported)",
    1831096: "GEG    Great Elm Group ($27B float reported)",
}

raw = pd.read_csv("scripts/step1_raw.csv")
extra = pd.read_csv(EXTRA_FILE).drop(columns="fy2024")

# 1. Candidates: step-1 companies that have a public float + the extras.
cands = pd.concat([raw[raw.public_float_usd.notna()], extra], ignore_index=True)

# 2. Fix known data errors.
fix = cands.cik.isin(FLOAT_DIVIDE_BY_1000)
cands.loc[fix, "public_float_usd"] = cands.loc[fix, "public_float_usd"] / 1000
cands = cands[~cands.cik.isin(EXCLUDE)]

# 3. Rank by public float (a proxy for company size), largest first.
cands["public_float_usd"] = cands.public_float_usd.astype("int64")  # truncates, as the original did
# kind="quicksort" (pandas' default) is what the original manual run used: it orders ties
# (equal floats, e.g. BSY/CVLT at $6.7B) the same way, so ranks match byte-for-byte.
# Ties never affect which companies make the top 130 (the cutoff falls between distinct values).
ranked = cands.sort_values("public_float_usd", ascending=False, kind="quicksort").reset_index(drop=True)
ranked.insert(0, "rank", range(1, len(ranked) + 1))
cols = ["rank", "cik", "ticker", "name", "sic", "n_10k_fy2015_2024", "public_float_usd"]
ranked = ranked[cols]

# 4. Keep the top N.
top = ranked.head(TOP_N)
ranked.to_csv("scripts/companies_all_ranked.csv", index=False)
top.to_csv("companies.csv", index=False)
with open("scripts/ciks.txt", "w") as f:  # input list for edgar-crawler
    f.write("".join(f"{c}\n" for c in top.cik))

print(f"candidates {len(raw[raw.public_float_usd.notna()])} + extras {len(extra)} "
      f"- excluded {len(EXCLUDE)} = ranked {len(ranked)}; kept top {len(top)}  (in {CORPUS_DIR})")
