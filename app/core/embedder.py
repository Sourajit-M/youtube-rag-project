from functools import lru_cache
from fastembed import TextEmbedding
from app.config import get_settings

class Embedder:
  """
    Wraps FastEmbed's TextEmbedding model.

    Why FastEmbed over sentence-transformers here:
    - ONNX runtime: no PyTorch dependency, ~200MB RAM vs ~2GB
    - Same model weights (all-MiniLM-L6-v2), identical vectors
    - Faster cold start — critical for API startup time
    - CPU-only: works on any free cloud tier

    The model produces 384-dimensional float vectors.
    Every chunk and every query gets embedded into this same 384-dim space
    so cosine similarity comparisons are valid.
  """
  MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

  def __init__(self):
    self.model = TextEmbedding(model_name = self.MODEL_NAME)

  def embed(self, texts: list[str]) -> list[list[float]]:
    """
        Embed a list of texts. Returns a list of 384-dim vectors.

        FastEmbed's embed() returns a generator of numpy arrays.
        We convert to plain Python lists so ChromaDB can serialise them
        without any numpy dependency in the DB layer.

        Batching: FastEmbed handles batching internally. Pass all chunks
        for a video in one call — more efficient than one call per chunk.
      """

    if not texts:
      return []
    
    embeddings = list(self.model.embed(texts))
    return [e.tolist() for e in embeddings]
  
  def embed_one(self, text: str) -> list[float]:
    return self.embed([text])[0]


@lru_cache
def get_embedder() -> Embedder:
  return Embedder()