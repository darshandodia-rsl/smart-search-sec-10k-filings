"""Shared paths and settings: the one place that knows where things live.

Every module imports locations from here instead of computing them from its own file
position, so moving a module never breaks a path. Secrets/settings come from .env.
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]          # project root (src/sec10k/config.py -> ../..)
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"                                 # git-ignored
CORPUS = DATA / "edgar-software-corpus"              # downloaded corpus (read-only)
RAW_10K = CORPUS / "data" / "raw" / "10-K"           # raw filing HTML
STATE_DIR = DATA / "state"                           # LLM daily-budget counters
PROFILE_DIR = DATA / "profile"                       # Phase 1 profiling outputs
HEADINGS_DIR = DATA / "headings"                     # cached emphasized blocks per filing

# Local embedding model. The chunker sizes chunks with this model's tokenizer, so both
# must always be the same model.
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
EMBED_DIM = 768

# The four 10-K sections in the corpus, in document order.
SECTIONS = ["item_1", "item_1A", "item_7", "item_7A"]


def load_manifest() -> pd.DataFrame:
    """The authoritative filing list (950 filings, FY2015-2024)."""
    return pd.read_csv(CORPUS / "manifest.csv", dtype={"cik": str})


# Neo4j (docker-compose.yml); the password lives only in .env.
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")


def load_companies() -> pd.DataFrame:
    """The 130 companies (rank, cik, ticker, name, sic, ...)."""
    return pd.read_csv(CORPUS / "companies.csv", dtype={"cik": str})
