"""Download the pinned 10-K filings from SEC EDGAR and turn HTML into plain text.

Two changes from the notebook, both for publishing:

1. The filings come from data/manifest_pinned.json, not from "the newest 3 10-Ks".
   "Newest 3" changes every time a company files a new 10-K (Apple files each
   autumn), which would silently change the corpus and break every gold chunk ID.
   Pinning the accession numbers makes the corpus reproducible forever.

2. The SEC contact e-mail is read from the SEC_USER_AGENT environment variable,
   so no personal e-mail address ends up in a public repository.
"""
import hashlib
import json
import os
import time

from bs4 import BeautifulSoup

from .config import EDGAR_CACHE, PINNED_MANIFEST

_last_call = [0.0]


def _user_agent() -> str:
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise RuntimeError(
            "SEC requires a User-Agent with contact details. Set it first, e.g.\n"
            '  export SEC_USER_AGENT="Your Name your.email@example.com"')
    return ua


def polite_get(url: str) -> str:
    """GET with a disk cache and a >=150 ms gap between live calls (SEC limit: 10/sec)."""
    import requests

    EDGAR_CACHE.mkdir(parents=True, exist_ok=True)
    cached = EDGAR_CACHE / (hashlib.md5(url.encode()).hexdigest() + ".txt")
    if cached.exists():                       # filings never change once published
        return cached.read_text(encoding="utf-8")

    gap = time.time() - _last_call[0]
    if gap < 0.15:
        time.sleep(0.15 - gap)
    _last_call[0] = time.time()

    r = requests.get(url, headers={"User-Agent": _user_agent()}, timeout=30)
    r.raise_for_status()
    cached.write_text(r.text, encoding="utf-8")
    return r.text


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def load_manifest() -> list[dict]:
    return json.loads(PINNED_MANIFEST.read_text(encoding="utf-8"))


def load_filings() -> list[dict]:
    """Return [{'source': 'AAPL_2025', 'text': ..., **manifest row}], one per filing.

    The manifest drives the loop. char_len is checked so a changed download
    raises instead of silently shifting every chunk boundary.
    """
    filings, seen = [], set()
    for row in load_manifest():
        text = html_to_text(polite_get(row["source_url"]))
        if len(text) != row["char_len"]:
            raise ValueError(
                f"{row['filename']}: expected {row['char_len']:,} chars, got {len(text):,}. "
                "BeautifulSoup version differences can cause this; see README.")
        source = row["filename"].removesuffix(".txt")
        if source in seen:
            raise ValueError(f"duplicate corpus entry: {source}")
        seen.add(source)
        filings.append({"source": source, "text": text, **row})
    return filings
