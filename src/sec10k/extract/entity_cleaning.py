"""Clean the raw GLiNER entities (tier 3) and store the result separately, per filing.

What it does:
  - `clean_all()` reads the raw GLiNER files in `data/entities/` (which it never changes),
    applies the cleaning rules and writes `data/entities_clean/{accession}.json`.
  - `load_entities()` returns a filing's cleaned entities per chunk; the rest of the pipeline
    reads entities only through this.
  - `names_other_company()` answers the route-B question for a chunk: does it name a company
    other than the filer?
  - `report()` prints summary numbers of the cleaned entities.

Why it is needed: GLiNER's raw output is noisy ("our products", "OEMs", the filer's own
abbreviations). Raw output is kept untouched for reference and so rules can change without
a new GPU run; the cleaned files are what later steps (LLM routing, graph writing) use.

The rules, in order:
  1. Drop non-names: text not starting with a capital letter or digit, numbers only, pronouns
     and common generic words ("We", "the Company"), securities as products ("common stock"),
     accounting standards ("Topic 606", "ASC 842"), and names made only of generic words
     ("Platform", "SaaS solutions", "Internet").
  2. Drop generic groups labeled as a company ("OEMs", "FinTechs", "Issuing Banks").
  3. Majority label per filer: if GLiNER gives the exact same name a different label more
     often across the filer's filings ("Tinder" mostly "product" in Match Group filings, "U.S."
     mostly a place), that label wins. The original label is kept as `raw_label`.
  4. Mark self-mentions (`is_self`): the filer's aliases and official name, its ticker, the
     first word of a name ("Manhattan" for Manhattan Associates), initials of the official
     name ("JHA" for Jack Henry & Associates), and the filer's own brands labeled as a
     company: names that start like the filer's name ("FortiGuard" ~ Fortinet) and whose
     company mentions come mostly (>= 80%) from that filer across the whole corpus.
     Self-mentions stay in the file but do not count as "another company".

Two passes: the first reads all raw files to count labels per filer (rule 3) and which filers
mention each company name (rule 4); the second cleans and writes. Each cleaned file stores
CLEAN_VERSION and a key made from the hashes of all raw files; everything is rebuilt when
either changes, otherwise skipped. Takes a few seconds. Bump CLEAN_VERSION when a rule changes.

Run: .venv/bin/python -m sec10k.extract.entity_cleaning clean [TICKER ...]
     .venv/bin/python -m sec10k.extract.entity_cleaning report [TICKER ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from sec10k.config import ENTITIES_CLEAN_DIR, ENTITIES_DIR, load_companies, load_manifest
from sec10k.extract.companies import CompanyMatcher, load_aliases
from sec10k.extract.entities import LABELS

CLEAN_VERSION = 2

# --------------------------------------------------------------------------------------
# Rule data
# --------------------------------------------------------------------------------------
# Whole names that are never useful entities.
_GENERIC_NAMES = {
    "we", "us", "our", "company", "the company", "companies", "sec", "notes", "covid-19",
    "board", "board of directors", "nasdaq", "nyse", "fasb", "gaap", "irs", "congress",
    "government", "u.s. government", "ai", "ai/ml", "retailers", "customers", "partners",
}
# Words that, when a name consists only of them, make it generic ("Cloud Platform", "Apps").
_GENERIC_WORDS = {
    "products", "product", "solutions", "solution", "services", "service", "platform",
    "platforms", "apps", "app", "applications", "application", "software", "cloud", "saas",
    "paas", "iaas", "api", "apis", "internet", "it", "ip", "iot", "ai", "ml", "technologies",
    "technology", "data", "network", "networks", "systems", "system", "tools", "offerings",
    "subscription", "subscriptions", "web", "mobile", "online", "digital", "enterprise",
    "customer", "customers", "partner", "partners", "users", "user", "developers", "the",
    "our", "and", "of", "new", "based", "cloud-based", "analytics", "security", "management",
    "marketplace", "infrastructure", "hardware", "devices", "device", "content", "media",
    "social", "email", "sms", "e-commerce", "ecommerce", "payments", "payment", "core",
    "server", "servers", "storage", "database", "databases", "open", "source", "suite",
}
# Last words that mark a group of companies, not one company ("Issuing Banks", "P&C insurers").
_GROUP_HEADS = {
    "banks", "insurers", "carriers", "providers", "partners", "customers", "merchants",
    "retailers", "developers", "users", "pinners", "fintechs", "issuers", "institutions",
    "vendors", "resellers", "oems", "isvs", "smbs", "msps", "rcfis", "fis", "advertisers",
    "publishers", "operators", "manufacturers", "suppliers", "distributors", "agencies",
    "governments", "clients", "companies", "enterprises", "brands", "creators", "sellers",
    "buyers", "hospitals", "payers", "lenders", "unions", "integrators", "competitors",
    # singular forms used as a group name ("RCFI", "FinTech")
    "bank", "insurer", "carrier", "fintech", "issuer", "oem", "isv", "smb", "msp", "rcfi", "fi",
}
# Legal-form words left out when building a company's initials ("JACK HENRY & ASSOCIATES INC").
_SUFFIX_WORDS = {"inc", "corp", "corporation", "co", "plc", "ltd", "llc", "holdings", "holding",
                 "group", "nv", "sa", "ag", "de", "ca", "ma"}
BRAND_SHARE = 0.8  # rule 4: share of a name's company mentions that must come from the filer
STEM_LEN = 5       # rule 4: how many first letters a brand must share with the filer's name
_ACRONYM_PLURAL = re.compile(r"^[A-Z]{2,}s$")                      # "OEMs", "SMBs"
_SECURITIES = re.compile(r"\b(common stock|stock|notes?|debentures|warrants?|shares)\b", re.I)
_ACCOUNTING = re.compile(r"^(ASC|ASU|Topic|IFRS|SFAS|FASB|SAB)\b")


def _norm(text: str) -> str:
    """Lowercase and keep only letters/digits, so "Digital Ocean, Inc." ~ "DigitalOcean"."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _words(text: str) -> list[str]:
    """Lowercased words of a name, split at spaces and hyphens, punctuation at the edges removed."""
    return [w.strip(".,;:()'’\"") for w in re.split(r"[\s-]+", text.lower()) if w.strip(".,;:()'’\"")]


@dataclass(frozen=True)
class Entity:
    """One cleaned entity found in a chunk."""

    text: str
    label: str            # final label, one of LABELS
    start: int            # character position in the chunk text
    end: int
    score: float          # GLiNER's confidence, 0-1
    is_self: bool         # True if it names the filer itself
    raw_label: str | None = None  # GLiNER's label, only set when rule 3 changed it


class Cleaner:
    """Applies the cleaning rules; knows each company's names to spot self-mentions.

    `company_filers` (from the first pass) maps an exact name to how often each filer's
    filings label it a company; rule 4 uses it to tell the filer's own brands from real
    companies that happen to start with the same letters.
    """

    def __init__(self, company_filers: dict[str, Counter] | None = None):
        self.matcher = CompanyMatcher()
        self.company_filers = company_filers or {}
        self.self_prefixes: dict[str, list[str]] = defaultdict(list)  # normalized names
        self.self_exact: dict[str, set[str]] = defaultdict(set)       # ticker, first words, initials
        self.self_stems: dict[str, set[str]] = defaultdict(set)       # first letters of the name
        names = [(r.ticker, r.alias) for r in load_aliases().itertuples()]
        names += [(r.ticker, r.name) for r in load_companies().itertuples()]
        for ticker, name in names:
            self.self_prefixes[ticker].append(_norm(name))
            self.self_exact[ticker].add(_norm(ticker))
            words = [w for w in re.split(r"[\s,/]+", name) if w and w.lower() != "the"]
            core = [w for w in words if _norm(w) and _norm(w) not in _SUFFIX_WORDS]
            if core:
                self.self_exact[ticker].add(_norm(core[0]))
                if len(_norm(core[0])) > STEM_LEN:
                    self.self_stems[ticker].add(_norm(core[0])[:STEM_LEN])
            initials = "".join(w[0] for w in core if w[0].isalpha() and w[0].isupper())
            if len(core) > 1 and len(initials) >= 3:
                self.self_exact[ticker].add(initials.lower())

    def _is_noise(self, text: str, label: str) -> bool:
        """Rules 1-2: True if the name is not a useful entity."""
        low, words = text.lower(), _words(text)
        if not re.match(r"[A-Z0-9]", text) or text.isdigit() or low in _GENERIC_NAMES:
            return True
        if low.startswith(("the company", "our ")) or _ACCOUNTING.match(text):
            return True
        if words and all(w in _GENERIC_WORDS for w in words):
            return True
        if label == "product" and _SECURITIES.search(text):
            return True
        if label == "company" and (_ACRONYM_PLURAL.match(text) or (words and words[-1] in _GROUP_HEADS)):
            return True
        return False

    def _is_self(self, text: str, ticker: str) -> bool:
        """Rule 4: True if the name refers to the filer itself."""
        n = _norm(text)
        if not n:
            return False
        if (n in self.self_exact[ticker]
                or any(m.is_self for m in self.matcher.find(text, ticker))
                or any(n.startswith(a) for a in self.self_prefixes[ticker] if a)):
            return True
        # Own brand: starts like the filer's name and is (almost) only called a company by this filer.
        filers = self.company_filers.get(text)
        return bool(filers and any(n.startswith(s) for s in self.self_stems[ticker])
                    and filers[ticker] >= BRAND_SHARE * sum(filers.values()))

    def clean(self, raw: list[dict], ticker: str, votes: dict[str, Counter] | None = None) -> list[Entity]:
        """Return the entities in `raw` that pass the rules, relabeled and with `is_self` set.

        `votes` maps an exact name to the label counts GLiNER gave it across the filer's
        filings (see `count_labels()`); without it rule 3 is skipped.
        """
        out = []
        for e in raw:
            text = e["text"].strip()
            label = e["label"]
            if votes and (v := votes.get(text)):
                top, n_top = v.most_common(1)[0]
                if top != label and n_top > v[label]:
                    label = top
            if self._is_noise(text, label):
                continue
            out.append(Entity(text, label, e["start"], e["end"], e["score"],
                              self._is_self(text, ticker), e["label"] if label != e["label"] else None))
        return out


def count_labels(raw_paths) -> tuple[dict[str, dict[str, Counter]], dict[str, Counter]]:
    """First pass over the raw files.

    Returns (votes, company_filers): votes[ticker][name] = label counts GLiNER gave the exact
    name in that filer's filings (rule 3); company_filers[name] = how often each filer labels
    it a company (rule 4).
    """
    votes: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    company_filers: dict[str, Counter] = defaultdict(Counter)
    for p in raw_paths:
        data = json.loads(p.read_text())
        for c in data["chunks"]:
            for e in c["entities"]:
                text = e["text"].strip()
                votes[data["ticker"]][text][e["label"]] += 1
                if e["label"] == "company":
                    company_filers[text][data["ticker"]] += 1
    return votes, company_filers


def names_other_company(entities: list[Entity]) -> bool:
    """True if the chunk names a company other than the filer (the tier-3 part of the route-B flag)."""
    return any(e.label == "company" and not e.is_self for e in entities)


# --------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------
def _clean_path(accession: str):
    """Where a filing's cleaned entities are stored."""
    return ENTITIES_CLEAN_DIR / f"{accession}.json"


def clean_all(tickers: list[str] | None = None) -> None:
    """Write cleaned files for all filings whose raw files exist (or only some companies').

    The first pass always reads every raw file, because rule 4 needs corpus-wide counts. The
    key stored in each cleaned file is CLEAN_VERSION + the hashes of all raw files; filings
    whose cleaned file already has the current key are skipped.
    """
    manifest = load_manifest()
    all_paths = [p for p in (ENTITIES_DIR / f"{a}.json" for a in manifest.accession_number) if p.exists()]
    key = hashlib.sha1((f"v{CLEAN_VERSION}|" + "|".join(
        sorted(hashlib.sha1(p.read_bytes()).hexdigest() for p in all_paths))).encode()).hexdigest()
    if tickers:
        manifest = manifest[manifest["ticker"].isin(tickers)]
    todo = [p for p in (ENTITIES_DIR / f"{a}.json" for a in manifest.accession_number) if p.exists()
            and not (_clean_path(p.stem).exists()
                     and json.loads(_clean_path(p.stem).read_text()).get("source_key") == key)]
    if not todo:
        print(f"All cleaned files up to date (clean_version {CLEAN_VERSION})")
        return
    votes, company_filers = count_labels(all_paths)
    cleaner = Cleaner(company_filers)
    ENTITIES_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    for p in todo:
        data = json.loads(p.read_text())
        ticker = data["ticker"]
        chunks = []
        for c in data["chunks"]:
            ents = cleaner.clean(c["entities"], ticker, votes[ticker])
            chunks.append({"chunk_id": c["chunk_id"], "content_hash": c["content_hash"],
                           "names_other_company": names_other_company(ents),
                           "entities": [{k: v for k, v in asdict(e).items() if v is not None}
                                        for e in ents]})
        clean = {"accession": data["accession"], "ticker": ticker,
                 "fiscal_year": data["fiscal_year"], "clean_version": CLEAN_VERSION,
                 "source_key": key, "raw_settings": data["settings"], "chunks": chunks}
        out = _clean_path(p.stem)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(clean))
        tmp.replace(out)  # atomic: the file is either complete or absent
    print(f"Cleaned {len(todo)} filings (clean_version {CLEAN_VERSION}); "
          f"{len(all_paths) - len(todo)} raw files have no work to do or were not selected")


def load_entities(accession: str) -> dict[str, list[Entity]]:
    """Return the cleaned entities of one filing, keyed by chunk_id."""
    p = _clean_path(accession)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `python -m sec10k.extract.entity_cleaning clean`")
    data = json.loads(p.read_text())
    return {c["chunk_id"]: [Entity(**e) for e in c["entities"]] for c in data["chunks"]}


def report(tickers: list[str] | None = None) -> None:
    """Print how many filings have cleaned files and summary numbers of the cleaned entities."""
    manifest = load_manifest()
    if tickers:
        manifest = manifest[manifest["ticker"].isin(tickers)]
    n_files = n_chunks = n_flag = n_relabeled = 0
    by_label, top = Counter(), {lab: Counter() for lab in LABELS}
    for accession in manifest.accession_number:
        p = _clean_path(accession)
        if not p.exists():
            continue
        n_files += 1
        for c in json.loads(p.read_text())["chunks"]:
            n_chunks += 1
            n_flag += c["names_other_company"]
            for e in c["entities"]:
                n_relabeled += "raw_label" in e
                if not e["is_self"]:
                    by_label[e["label"]] += 1
                    top[e["label"]][e["text"]] += 1
    print(f"Cleaned filings: {n_files} / {len(manifest)}  ({n_chunks:,} chunks)")
    if not n_chunks:
        return
    print(f"Chunks naming a company other than the filer: {n_flag:,} ({n_flag / n_chunks:.1%})")
    print(f"Entities relabeled by the per-filer majority rule: {n_relabeled:,}")
    for lab in LABELS:
        print(f"  {lab}: {by_label[lab]:,} non-self entities; most common: "
              + ", ".join(f"{t} {n}" for t, n in top[lab].most_common(10)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["clean", "report"])
    ap.add_argument("tickers", nargs="*", help="only these companies (default: all)")
    a = ap.parse_args()
    if a.command == "clean":
        clean_all(a.tickers or None)
    report(a.tickers or None)
