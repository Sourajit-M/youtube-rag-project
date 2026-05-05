from dataclasses import dataclass
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings

@dataclass
class ChunkResult:
  chunk_id: str
  text: str
  video_youtube_id: str
  video_title: str
  channel_name: str
  chunk_index: int
  distance: float

  @property
  def similarity(self) -> float:
    """Convert distance to similarity score 0.0-1.0 for display."""
    return round(1 - self.distance, 4)

class VectorDB:
  COLLECTION_NAME = "video_chunks"

  def __init__(self):
    settings = get_settings()
    self._client = chromadb.PersistentClient(
      path=str(settings.chroma_db_path),
      settings=ChromaSettings(anonymized_telemetry=False),
    )
    self._collection = self._client.get_or_create_collection(
      name=self.COLLECTION_NAME,
      metadata={"hnsw:space": "cosine"},  # distance metric set at creation
    )

  #write
  def upsert_chunks(self, chunks: list[str], embeddings: list[list[float]], video_youtube_id: str,video_title: str, channel_name: str,) -> int:
    """
    Store chunks for one video. Returns number of chunks stored.

    'upsert' = insert or replace. If the video was already ingested,
    calling this again replaces its chunks cleanly. Idempotent.

    chunk_index in metadata lets us reconstruct reading order if needed.
    It also helps with debugging ('chunk 34 of video X was retrieved').
    """
    if not chunks:
      return 0

    ids = [f"{video_youtube_id}_chunk_{i}" for i in range(len(chunks))]

    metadatas = [
      {
        "video_youtube_id": video_youtube_id,
        "video_title": video_title,
        "channel_name": channel_name,
        "chunk_index": i,
      }
      for i in range(len(chunks))
    ]

    self._collection.upsert(
      ids=ids,
      documents=chunks,
      embeddings=embeddings,
      metadatas=metadatas,
    )

    return len(chunks)

  def delete_video_chunks(self, video_youtube_id: str) -> None:
    """
    Remove all chunks for a video.
    Useful when a video is deleted from a channel or re-ingested from scratch.
    """
    self._collection.delete(
      where={"video_youtube_id": video_youtube_id}
    )

  # ── Read

  def search(self, query_embedding: list[float], top_k: int, channel_name: Optional[str] = None) -> list[ChunkResult]:
    """
    Vector similarity search. Returns top_k most similar chunks.

    channel_name filter lets users search within one channel only.
    ChromaDB's 'where' clause applies the filter before ranking —
    efficient, not a post-filter.

    Note: this is called by the hybrid retriever alongside BM25.
    The retriever fuses both result lists — this method only does
    the vector half.
    """
    where = {"channel_name": channel_name} if channel_name else None

    results = self._collection.query(
      query_embeddings=[query_embedding],
      n_results=top_k,
      where=where,
      include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["ids"][0])):
      meta = results["metadatas"][0][i]
      chunks.append(ChunkResult(
        chunk_id=results["ids"][0][i],
        text=results["documents"][0][i],
        video_youtube_id=meta["video_youtube_id"],
        video_title=meta["video_title"],
        channel_name=meta["channel_name"],
        chunk_index=meta["chunk_index"],
        distance=results["distances"][0][i],
      ))

    return chunks

  def get_all_chunks_for_bm25(self, channel_name : Optional[str] = None) -> tuple[list[str], list[str], list[dict]]:
    """
    Returns ALL chunk texts, ids, and metadatas for building the BM25 index.

    BM25 is a separate in-memory index — it needs the raw text of every
    chunk, not embeddings. We load this once at startup and rebuild it
    when new videos are ingested.

    Returns: (texts, ids, metadatas)
    """
    where = {"channel_name": channel_name} if channel_name else None

    results = self._collection.get(
      where=where,
      include=["documents", "metadatas"],
    )

    return (
      results["documents"],
      results["ids"],
      results["metadatas"],
    )

  #Stats

  def count(self) -> int:
    """Total number of chunks stored."""
    return self._collection.count()

  def stats(self) -> dict:
    """Summary stats — used by the /health and /stats endpoints."""
    return {
      "collection": self.COLLECTION_NAME,
      "total_chunks": self.count(),
    }