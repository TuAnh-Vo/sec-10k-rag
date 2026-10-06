# artifacts/ — the finished index

Put these files here (from the notebook's export cell, section 8b):

| File | What it is |
|---|---|
| `chunks.jsonl` | 5,089 lines, one chunk each: `chunk_id`, `text`, `meta` |
| `embeddings.npy` | float32 matrix `[5089, 768]`, unit-length rows; row *i* = line *i* |
| `info.json` | count, dimension, embedding model name |

`rag/store.py` checks on load that the counts match, the IDs are unique and every
vector has length 1, so a mismatched pair of files fails loudly.
