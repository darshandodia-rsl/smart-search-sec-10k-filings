"""Shared paths and settings for the whole project.

What it does: defines where the data lives (corpus, caches, state files), which embedding
model is used, which 10-K sections we handle, and how to connect to Neo4j. It also loads
the secrets in `.env`.

Why it is needed: every other module (loader, chunker, embedder, graph writers, LLM
backends) imports locations and settings from here instead of working them out itself.
That keeps all settings in one place, so changing one (say, the embedding model) changes
it everywhere at once.
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Project root folder. This file is src/sec10k/config.py, so the root is two levels up.
ROOT = Path(__file__).resolve().parents[2]
# Read secrets and overrides (like the Neo4j password) from .env into environment variables.
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"                                 # all generated data; not in git
CORPUS = DATA / "edgar-software-corpus"              # the downloaded 10-K corpus (read-only)
RAW_10K = CORPUS / "data" / "raw" / "10-K"           # the original filing HTML files
STATE_DIR = DATA / "state"                           # LLM daily token-budget counters
PROFILE_DIR = DATA / "profile"                       # output of the corpus profiling scripts
HEADINGS_DIR = DATA / "headings"                     # cached bold/italic blocks per filing
ENTITIES_DIR = DATA / "entities"                     # raw GLiNER entities per filing (tier 3, never edited)
ENTITIES_CLEAN_DIR = DATA / "entities_clean"         # cleaned entities per filing, rebuilt from raw
# Hand-checked names each corpus company goes by (in git, next to the code that uses it).
COMPANY_ALIASES = ROOT / "src" / "sec10k" / "extract" / "company_aliases.csv"

# The local embedding model and the size of the vectors it produces. The chunker measures
# chunk sizes with this model's tokenizer, so chunker and embedder must use the same model.
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
EMBED_DIM = 768

# The four 10-K sections we use, in the order they appear in a filing:
# Business, Risk Factors, MD&A, Market Risk.
SECTIONS = ["item_1", "item_1A", "item_7", "item_7A"]


def load_manifest() -> pd.DataFrame:
    """Return the official list of filings to process (950 rows, fiscal years 2015-2024).

    Every pipeline step picks its filings from this list, so they all work on the same set.
    """
    return pd.read_csv(CORPUS / "manifest.csv", dtype={"cik": str})


# Neo4j connection settings (the database runs in Docker, see docker-compose.yml).
# The password is only ever read from .env, never written in code.
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")


def load_companies() -> pd.DataFrame:
    """Return the 130 companies in the corpus (rank, cik, ticker, name, sic, ...).

    Used to create the Company nodes in the graph, and later to recognise company names
    mentioned inside filings.
    """
    return pd.read_csv(CORPUS / "companies.csv", dtype={"cik": str})
