"""Every tunable number in one place.

These values are the ones used to produce the results in the README.
Change one, and the chunk IDs (and therefore the eval set's gold IDs) may change too.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- data ------------------------------------------------------------------
PINNED_MANIFEST = ROOT / "data" / "manifest_pinned.json"   # the exact 15 filings
EDGAR_CACHE = ROOT / "edgar_cache"                         # raw HTML, git-ignored
ARTIFACTS = ROOT / "artifacts"                             # chunks + vectors (committed)

# --- chunking (section 3-5 of the notebook) --------------------------------
CHUNK_CHARS = 1100
CHUNK_OVERLAP = 200
SEC_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
MIN_CHARS = 200

# Metadata that rides along with every chunk.
META_KEYS = ("ticker", "company", "cik", "fiscal_year",
             "report_date", "filing_date", "source_url", "accession")

# --- retrieval ---------------------------------------------------------------
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATE_K = 50
RRF_K = 60
MMR_LAMBDA = 0.95

# --- generation --------------------------------------------------------------
GENERATOR_NAME = "Qwen/Qwen3-8B"
MAX_NEW_TOKENS = 300
TOP_K_CONTEXT = 5
