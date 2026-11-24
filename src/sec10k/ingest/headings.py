"""Phase 2.2a: recover heading formatting (bold / italic / underline) from the raw HTML.

The crawler's HTML-to-text step dropped all formatting, but 10-Ks mark their structure with
it: risk titles in bold or italic, risk groups underlined, subsections bold. The chunker
uses these lines as chunk boundaries, so a chunk never spans two risk factors.

We don't re-extract the corpus (that would move section boundaries and void Phase 1).
Instead, a side pass parses each raw .htm once and records the *text* of emphasized
blocks. The chunker then looks lines up by normalized text.

Per block (p / div / li / h1-6 without nested blocks, outside tables) we record:
  "full": every character is emphasized              -> a heading or risk title
  "lead": the block starts with an emphasized phrase -> run-in title ("Competition. We ...")
Emphasis comes from tags (b, strong, i, em, u) or inline CSS (font-weight >= 600,
font-style: italic, text-decoration: underline); the closest ancestor that sets a property
wins, so <span style="font-weight:normal"> inside a bold block counts as not bold.
Not handled: emphasis set via CSS classes in a <style> block (see coverage report).

Output: data/headings/<filing>.json  (derived, git-ignored, one per manifest filing)
Run from the project root:  .venv/bin/python -m sec10k.ingest.headings
"""

import json
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lxml.html

from sec10k.config import HEADINGS_DIR as OUT
from sec10k.config import RAW_10K as RAW
from sec10k.config import load_manifest
BLOCKS = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"}
KEY_CHARS = 100          # lines are matched on the first 100 normalized characters
MAX_FULL_CHARS = 600     # a fully emphasized block longer than this is not a heading

TAG_STYLE = {"b": "bold", "strong": "bold", "i": "italic", "em": "italic", "u": "underline"}
CSS_RULES = {
    "bold": re.compile(r"font-weight\s*:\s*([a-z0-9]+)", re.I),
    "italic": re.compile(r"font-style\s*:\s*([a-z]+)", re.I),
    "underline": re.compile(r"text-decoration(?:-line)?\s*:\s*([a-z ]+)", re.I),
}


def css_value(prop: str, value: str) -> bool:
    v = value.lower()
    if prop == "bold":
        return v in {"bold", "bolder"} or (v.isdigit() and int(v) >= 600)
    if prop == "italic":
        return v in {"italic", "oblique"}
    return "underline" in v


def norm(text: str) -> str:
    """Normalize so HTML text and the crawler's extracted text compare equal."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"[‘’`´]", "'", t)
    t = re.sub(r"[“”]", '"', t)
    t = re.sub(r"[‐-―−]", "-", t)
    t = re.sub(r"[​\xa0]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def key(text: str) -> str:
    return norm(text)[:KEY_CHARS]


def styles_of(el, block) -> frozenset[str]:
    """Emphasis styles in effect for text directly inside `el` (walking up to `block`)."""
    decided: dict[str, bool] = {}
    node = el
    while node is not None:
        tag = node.tag if isinstance(node.tag, str) else ""
        tag = tag.split(":")[-1].lower()
        if tag in TAG_STYLE and TAG_STYLE[tag] not in decided:
            decided[TAG_STYLE[tag]] = True
        style = node.get("style") or ""
        for prop, rx in CSS_RULES.items():
            if prop not in decided:
                m = rx.search(style)
                if m:
                    decided[prop] = css_value(prop, m.group(1).strip())
        if node is block or len(decided) == 3:
            break
        node = node.getparent()
    return frozenset(p for p, on in decided.items() if on)


def text_runs(block):
    """Yield (text, styles) for each piece of text in the block, in document order:
    an element's own text, then each child's subtree followed by that child's tail."""
    def walk(el):
        if not isinstance(el.tag, str):   # <!-- comment --> / <?pi?>: its .text is not visible
            return
        if el.text:
            yield el.text, styles_of(el, block)
        for child in el:
            yield from walk(child)
            if child.tail:
                yield child.tail, styles_of(el, block)
    yield from walk(block)


def emphasized_blocks(html_path: Path) -> list[dict]:
    doc = lxml.html.fromstring(html_path.read_bytes())
    out, seen = [], set()
    for block in doc.iter(*BLOCKS):
        if any(d.tag in BLOCKS for d in block.iterdescendants() if isinstance(d.tag, str)):
            continue                                   # not a leaf block
        if any(a.tag == "table" for a in block.iterancestors()):
            continue                                   # table cells: column headers, totals
        runs = [(t, s) for t, s in text_runs(block) if t.strip()]
        if not runs:
            continue
        full_text = norm("".join(t for t, _ in runs))
        if not re.search(r"[a-z]{2}", full_text):
            continue                                   # page numbers, symbols
        lead, lead_styles = [], set()
        for t, s in runs:
            if not s:
                break
            lead.append(t)
            lead_styles |= s
        lead_text = norm("".join(lead))
        if not lead_text:
            continue
        kind = "full" if len(lead) == len(runs) else "lead"
        if kind == "full" and len(full_text) > MAX_FULL_CHARS:
            continue                                   # whole emphasized paragraph, not a title
        if kind == "lead" and len(re.findall(r"[a-z]+", lead_text)) < 2:
            continue                                   # a single bold word, e.g. "Not"
        k = full_text[:KEY_CHARS]
        if k in seen:
            continue
        seen.add(k)
        out.append({"key": k, "kind": kind, "styles": "".join(sorted(x[0] for x in lead_styles)),
                    "lead": lead_text[:KEY_CHARS], "n_chars": len(full_text)})
    return out


def process(json_path: str) -> tuple[str, int, str | None]:
    name = Path(json_path).stem
    dest = OUT / f"{name}.json"
    if dest.exists():                                  # resumable: done files are skipped
        return name, -1, None
    try:
        blocks = emphasized_blocks(RAW / f"{name}.htm")
    except Exception as e:                             # keep going; report at the end
        return name, 0, repr(e)
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(blocks))
    tmp.rename(dest)                                   # atomic: no half-written cache files
    return name, len(blocks), None


def load_headings(json_path: str) -> dict[str, dict]:
    """key(line) -> block info, for one filing (by its manifest json_path)."""
    blocks = json.loads((OUT / f"{Path(json_path).stem}.json").read_text())
    return {b["key"]: b for b in blocks}


def lookup(headings: dict[str, dict], line: str) -> dict | None:
    """The emphasized block this line comes from, or None. Matches on the first KEY_CHARS
    characters *and* the total length: a body paragraph that opens with the same words as
    the risk title above it must not count as a heading. Lengths may differ slightly where
    the loader rejoined a page-split sentence, hence the small tolerance."""
    b = headings.get(key(line))
    if b and abs(len(norm(line)) - b["n_chars"]) <= 3:
        return b
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = load_manifest()["json_path"].tolist()
    done = skipped = 0
    errors = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        for i, (name, n, err) in enumerate(pool.map(process, paths, chunksize=4), 1):
            if err:
                errors.append((name, err))
            elif n < 0:
                skipped += 1
            else:
                done += 1
            if i % 100 == 0:
                print(f"  {i}/{len(paths)}", flush=True)
    print(f"Parsed {done}, already cached {skipped}, errors {len(errors)}")
    for name, err in errors:
        print("  ERROR", name, err)


if __name__ == "__main__":
    main()
