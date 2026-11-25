"""The local embedding model that turns text into vectors.

What it does: loads the bge-base-en-v1.5 model once and hands it to whoever needs it. The
model runs on the Mac's GPU (called MPS) and costs nothing per call.

Why it is needed: search by meaning works by comparing vectors. Chunks are embedded when
we ingest them, and user questions are embedded when we search. Both sides must use this
same model, or their vectors cannot be compared.

Usage note: bge adds a special instruction in front of questions ("Represent this question
for searching relevant passages: ") but not in front of documents. So embed chunks with
`get_text_embedding_batch()` and questions with `get_query_embedding()`.
"""

from functools import cache

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from sec10k.config import EMBED_MODEL

# The most tokens bge can read at once. The chunker keeps every chunk below this.
BGE_MAX_TOKENS = 512


@cache
def get_embed_model(batch_size: int = 32) -> HuggingFaceEmbedding:
    """Return the embedding model, loading it on the first call and reusing it after that.

    Loading takes a few seconds and about 440 MB of memory, so we do it once per process.
    The vectors are normalized to length 1, which means cosine similarity (the score used
    to rank search results) equals a plain dot product.
    """
    return HuggingFaceEmbedding(model_name=EMBED_MODEL, device="mps", max_length=BGE_MAX_TOKENS,
                                normalize=True, embed_batch_size=batch_size)
