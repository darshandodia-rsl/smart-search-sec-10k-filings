"""The local embedding model, shared by ingestion (chunks) and retrieval (queries).

bge-base-en-v1.5 runs on the M3 GPU (MPS). sentence-transformers applies bge's prompts
itself: queries get "Represent this question for searching relevant passages: ", documents
get none. So always embed chunks with get_text_embedding_batch() and questions with
get_query_embedding(), never the other way round.
"""

from functools import cache

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from sec10k.config import EMBED_MODEL

BGE_MAX_TOKENS = 512       # bge's input limit; the chunker keeps every chunk under it


@cache
def get_embed_model(batch_size: int = 32) -> HuggingFaceEmbedding:
    """Loaded once per process (~440 MB). Vectors are L2-normalized (cosine = dot product)."""
    return HuggingFaceEmbedding(model_name=EMBED_MODEL, device="mps", max_length=BGE_MAX_TOKENS,
                                normalize=True, embed_batch_size=batch_size)
