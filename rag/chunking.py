"""Split filings into chunks, freeze chunk IDs, then remove junk.

The ORDER is the important part (it fixed the FAISS/BM25 desync bug):

    split -> MIN_CHARS filter -> assign chunk_id ONCE -> content filters

IDs are assigned before the content filters, so editing a filter leaves gaps in
the numbering instead of renumbering everything. Saved gold IDs stay valid.
"""
import re
from collections import defaultdict

from .config import CHUNK_CHARS, CHUNK_OVERLAP, META_KEYS, MIN_CHARS, SEC_SEPARATORS


def split_filings(filings: list[dict]) -> list[dict]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_CHARS, chunk_overlap=CHUNK_OVERLAP,
        separators=SEC_SEPARATORS, strip_whitespace=True)

    chunks = []
    for f in filings:
        meta = {"source": f["source"], **{k: f[k] for k in META_KEYS}}
        for piece in splitter.split_text(f["text"]):
            chunks.append({"text": piece, "meta": dict(meta)})
    return chunks


def assign_ids(chunks: list[dict]) -> list[dict]:
    """Readable, stable IDs like AAPL_2025__0148. Run exactly once."""
    counter = defaultdict(int)
    for c in chunks:
        src = c["meta"]["source"]
        c["chunk_id"] = f"{src}__{counter[src]:04d}"
        counter[src] += 1
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "duplicate chunk_id"
    return chunks


def is_junk(text: str) -> bool:
    """Machine-generated metadata: XBRL tag soup and digit-dense table fragments."""
    if len(re.findall(r"\b(us-gaap|dei|srt|msft|nvda|aapl|amzn|goog):", text)) >= 5:
        return True
    if sum(c.isalpha() for c in text) / max(len(text), 1) < 0.55:
        return True
    return False


def is_front_matter(text: str) -> bool:
    """Human-written administrative pages: signatures, exhibit indexes, TOC, cover."""
    t = text
    if t.count("/ S /") >= 2 or "POWER OF ATTORNEY" in t.upper():
        return True
    if re.search(r"/s/\s*[A-Z]", t) and "SIGNATURES" in t.upper():
        return True
    # Count FORM TYPES, not decimals: "10.8" looks like "$3.5 billion".
    if len(re.findall(r"\b(8-K|10-Q|10-K)\b", t)) >= 4:
        return True
    if "Index to Exhibits" in t or "Exhibit Number" in t:
        return True
    if len(re.findall(r"Item\s+\d+[A-C]?\.", t)) >= 5:
        return True
    if "DOCUMENTS INCORPORATED BY REFERENCE" in t.upper():
        return True
    if t.count("Indicate by check mark") >= 2:
        return True
    return False


def build_chunks(filings: list[dict], verbose: bool = True) -> list[dict]:
    chunks = split_filings(filings)
    n0 = len(chunks)
    chunks = [c for c in chunks if len(c["text"]) >= MIN_CHARS]
    n1 = len(chunks)
    chunks = assign_ids(chunks)                    # <- frozen here
    chunks = [c for c in chunks if not is_junk(c["text"])]
    n2 = len(chunks)
    chunks = [c for c in chunks if not is_front_matter(c["text"])]
    if verbose:
        print(f"split {n0:,} | after MIN_CHARS {n1:,} | after junk {n2:,} | "
              f"after front-matter {len(chunks):,}")
    return chunks
