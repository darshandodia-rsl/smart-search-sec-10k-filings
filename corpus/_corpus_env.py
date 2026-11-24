"""Shared setup for the corpus-build scripts (imported first by each one).

- loads the repo's .env (SEC_USER_AGENT lives there, never in git)
- switches the working directory to the corpus folder, so the scripts' relative paths
  (scripts/..., data/..., companies.csv) work no matter where you run them from.
  Override the folder with CORPUS_DIR=... (used to test the scripts on a scratch copy).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")

CORPUS_DIR = Path(os.getenv("CORPUS_DIR", REPO / "data" / "edgar-software-corpus"))
(CORPUS_DIR / "scripts").mkdir(parents=True, exist_ok=True)  # working files + HTTP cache
os.chdir(CORPUS_DIR)


def user_agent() -> str:
    """SEC requires a User-Agent with your name + email: https://www.sec.gov/os/accessing-edgar-data"""
    return os.environ["SEC_USER_AGENT"]
