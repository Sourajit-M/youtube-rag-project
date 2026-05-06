import pytest
from app.core.retriever import BM25Index, reciprocal_rank_fusion
from app.db.vectordb import ChunkResult


@pytest.fixture
def bm25_index():
  index = BM25Index()
  texts = [
    "photosynthesis converts sunlight into glucose in plant cells",
    "chlorophyll absorbs red and blue light reflects green",
    "CRISPR cas9 is a gene editing tool derived from bacteria",
    "Jennifer Doudna won the Nobel Prize for CRISPR research",
    "the Calvin cycle fixes carbon dioxide into sugar molecules",
  ]
  ids = [f"chunk_{i}" for i in range(len(texts))]
  metadatas = [
    {
      "video_youtube_id": f"vid_{i}",
      "video_title": f"Video {i}",
      "channel_name": "Test",
      "chunk_index": i,
    }
    for i in range(len(texts))
  ]
  index.build(texts=texts, chunk_ids=ids, metadatas=metadatas)
  return index


def test_bm25_returns_results(bm25_index):
  results = bm25_index.search("photosynthesis", top_k=3)
  assert len(results) > 0


def test_bm25_exact_term_ranks_first(bm25_index):
  results = bm25_index.search("CRISPR Nobel Prize", top_k=5)
  top_ids = [r[0] for r in results[:2]]

  # chunks 2 and 3 contain CRISPR/Nobel terms
  assert "chunk_2" in top_ids or "chunk_3" in top_ids


def test_bm25_no_match_returns_empty(bm25_index):
  results = bm25_index.search("quantum mechanics black holes", top_k=3)

  # BM25 returns 0.0 scores for no term overlap — all filtered out
  assert len(results) == 0


def test_bm25_channel_filter(bm25_index):
  results = bm25_index.search(
    "photosynthesis",
    top_k=3,
    channel_name="NonExistent",
  )
  assert len(results) == 0


def test_rrf_fusion_boosts_dual_ranked():
  """A chunk in both lists should outscore one in only one list."""
  bm25_results = [
    (
      "chunk_A",
      10.0,
      "text A",
      {
        "video_youtube_id": "v1",
        "video_title": "V1",
        "channel_name": "C",
        "chunk_index": 0,
      },
    ),
    (
      "chunk_B",
      5.0,
      "text B",
      {
        "video_youtube_id": "v2",
        "video_title": "V2",
        "channel_name": "C",
        "chunk_index": 0,
      },
    ),
  ]

  vector_results = [
    ChunkResult(
      chunk_id="chunk_B",
      text="text B",
      video_youtube_id="v2",
      video_title="V2",
      channel_name="C",
      chunk_index=0,
      distance=0.1,
    ),
    ChunkResult(
      chunk_id="chunk_C",
      text="text C",
      video_youtube_id="v3",
      video_title="V3",
      channel_name="C",
      chunk_index=0,
      distance=0.2,
    ),
  ]

  fused = reciprocal_rank_fusion(
    bm25_results,
    vector_results,
    top_k=3,
  )

  # chunk_B appears in both lists — should rank highest
  assert fused[0]["chunk_id"] == "chunk_B"


def test_rrf_top_k_respected():
  bm25_results = [
    (
      f"chunk_{i}",
      float(10 - i),
      f"text {i}",
      {
        "video_youtube_id": f"v{i}",
        "video_title": f"V{i}",
        "channel_name": "C",
        "chunk_index": i,
      },
    )
    for i in range(5)
  ]

  fused = reciprocal_rank_fusion(
    bm25_results,
    [],
    top_k=3,
  )

  assert len(fused) <= 3