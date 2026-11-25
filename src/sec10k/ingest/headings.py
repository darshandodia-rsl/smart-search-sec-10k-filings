"""Finds which lines of a filing are headings, using the bold/italic/underline in the original HTML.

What it does: reads each raw filing (.htm) once and records the text of every emphasized
block, saving the result in `data/headings/<filing>.json`. The chunker later looks each
line of text up in this cache.

Why it is needed: 10-Ks show their structure with formatting: risk titles are bold or
italic, risk groups are underlined, subsections are bold. The extracted text we work with
has lost all formatting, so without this step we could not tell a heading from a normal
paragraph. The chunker uses headings as chunk boundaries so that a chunk never spans two
risk factors, and each chunk can carry its heading.

The extracted corpus is left untouched. Re-extracting it could move section boundaries, so
instead this module reads the raw HTML on the side and matches by normalized text.

For each block (p / div / li / h1-h6 without nested blocks, outside tables) it records:
  "full": every character is emphasized         -> a heading or risk title
  "lead": the block starts with an emphasized phrase -> a run-in title ("Competition. We ...")
Emphasis comes from tags (b, strong, i, em, u) or inline CSS (font-weight 600 or more,
font-style italic, text-decoration underline). The closest element that sets a style wins.
Emphasis set through CSS classes in a <style> block is not detected.

Output: data/headings/<filing>.json (derived data, not in git, one file per filing).
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
# HTML tags that count as a block of text (a paragraph, list item, heading or div).
BLOCKS = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"}
KEY_CHARS = 100          # a line is identified by its first 100 normalized characters
MAX_FULL_CHARS = 600     # a fully emphasized block longer than this is a bold paragraph, not a heading

# HTML tags that switch an emphasis style on.
TAG_STYLE = {"b": "bold", "strong": "bold", "i": "italic", "em": "italic", "u": "underline"}
# Patterns that read an emphasis style out of an inline CSS style attribute.
CSS_RULES = {
    "bold": re.compile(r"font-weight\s*:\s*([a-z0-9]+)", re.I),
    "italic": re.compile(r"font-style\s*:\s*([a-z]+)", re.I),
    "underline": re.compile(r"text-decoration(?:-line)?\s*:\s*([a-z ]+)", re.I),
}


def css_value(prop: str, value: str) -> bool:
    """Say whether a CSS value turns the style on (e.g. font-weight 700 = bold, normal = not bold)."""
    v = value.lower()
    if prop == "bold":
        return v in {"bold", "bolder"} or (v.isdigit() and int(v) >= 600)
    if prop == "italic":
        return v in {"italic", "oblique"}
    return "underline" in v


def norm(text: str) -> str:
    """Normalize text (case, quotes, dashes, spaces) so the HTML text and the extracted text compare equal."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"[‘’`´]", "'", t)
    t = re.sub(r"[“”]", '"', t)
    t = re.sub(r"[‐-―−]", "-", t)
    t = re.sub(r"[​\xa0]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def key(text: str) -> str:
    """The short identifier of a line: its first KEY_CHARS characters, normalized."""
    return norm(text)[:KEY_CHARS]


def styles_of(el, block) -> frozenset[str]:
    """Return the emphasis styles (bold, italic, underline) in effect for text inside `el`.

    Walks up from `el` to `block`. The closest element that sets a style decides it, so a
    span with font-weight normal inside a bold block counts as not bold.
    """
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
    """Yield (text, styles) for each piece of text in the block, in reading order.

    Splitting the block into styled pieces is how we tell whether a block is fully
    emphasized or only starts with an emphasized phrase.
    """
    def walk(el):
        if not isinstance(el.tag, str):   # HTML comments and similar: their text is not visible
            return
        if el.text:
            yield el.text, styles_of(el, block)
        for child in el:
            yield from walk(child)
            if child.tail:
                yield child.tail, styles_of(el, block)
    yield from walk(block)


def emphasized_blocks(html_path: Path) -> list[dict]:
    """Find the heading-like blocks in one filing's HTML and describe each one.

    Returns one record per block: its identifying key, its kind ("full" or "lead", see the
    module docstring), the styles used, the emphasized lead text and the block's length.
    Skips table cells, page numbers, bold paragraphs and single bold words.
    """
    doc = lxml.html.fromstring(html_path.read_bytes())
    out, seen = [], set()
    for block in doc.iter(*BLOCKS):
        if any(d.tag in BLOCKS for d in block.iterdescendants() if isinstance(d.tag, str)):
            continue                                   # skip blocks that contain other blocks
        if any(a.tag == "table" for a in block.iterancestors()):
            continue                                   # skip table cells (column headers, totals)
        runs = [(t, s) for t, s in text_runs(block) if t.strip()]
        if not runs:
            continue
        full_text = norm("".join(t for t, _ in runs))
        if not re.search(r"[a-z]{2}", full_text):
            continue                                   # skip page numbers and symbols
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
            continue                                   # a fully bold paragraph is not a title
        if kind == "lead" and len(re.findall(r"[a-z]+", lead_text)) < 2:
            continue                                   # a single bold word is not a title
        k = full_text[:KEY_CHARS]
        if k in seen:
            continue
        seen.add(k)
        out.append({"key": k, "kind": kind, "styles": "".join(sorted(x[0] for x in lead_styles)),
                    "lead": lead_text[:KEY_CHARS], "n_chars": len(full_text)})
    return out


def process(json_path: str) -> tuple[str, int, str | None]:
    """Build and save the heading cache for one filing.

    Returns (filing name, number of blocks found, error text or None). A block count of -1
    means the cache file already existed and was skipped. The file is written to a temporary
    name and then renamed, so a crash never leaves a half-written cache file.
    """
    name = Path(json_path).stem
    dest = OUT / f"{name}.json"
    if dest.exists():                                  # already cached: skip, so reruns are cheap
        return name, -1, None
    try:
        blocks = emphasized_blocks(RAW / f"{name}.htm")
    except Exception as e:                             # one bad filing should not stop the rest
        return name, 0, repr(e)
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(blocks))
    tmp.rename(dest)                                   # the rename is atomic
    return name, len(blocks), None


def load_headings(json_path: str) -> dict[str, dict]:
    """Load the cached heading blocks of one filing, as a dict from line key to block info.

    `json_path` is the filing's path from the manifest. The chunker calls this once per filing.
    """
    blocks = json.loads((OUT / f"{Path(json_path).stem}.json").read_text())
    return {b["key"]: b for b in blocks}


def lookup(headings: dict[str, dict], line: str) -> dict | None:
    """Return the emphasized block this line comes from, or None if it is plain text.

    A line matches when its first KEY_CHARS characters AND its total length match. The
    length check stops a body paragraph that begins with the same words as its risk title
    from being mistaken for the title. A small length tolerance allows for sentences the
    loader rejoined after a page break.
    """
    b = headings.get(key(line))
    if b and abs(len(norm(line)) - b["n_chars"]) <= 3:
        return b
    return None


def main() -> None:
    """Build the heading cache for every filing in the manifest, in parallel, and print a summary."""
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
