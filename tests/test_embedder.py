import pytest
from app.core.embedder import Embedder, get_embedder


@pytest.fixture(scope="module")  # load model once for all tests in this file
def embedder():
  return Embedder()


def test_embed_returns_correct_dimensions(embedder):
  vectors = embedder.embed(["hello world"])
  assert len(vectors) == 1
  assert len(vectors[0]) == 384


def test_embed_batch(embedder):
  texts = ["first sentence", "second sentence", "third sentence"]
  vectors = embedder.embed(texts)
  assert len(vectors) == 3
  assert all(len(v) == 384 for v in vectors)


def test_embed_one_returns_single_vector(embedder):
  vec = embedder.embed_one("test query")
  assert len(vec) == 384
  assert isinstance(vec, list)


def test_embed_empty_list(embedder):
  result = embedder.embed([])
  assert result == []


def test_similar_texts_closer_than_unrelated(embedder):
  """Core sanity check — semantic similarity works."""
  vec_a = embedder.embed_one("photosynthesis converts sunlight to energy")
  vec_b = embedder.embed_one("plants use light to produce glucose")
  vec_c = embedder.embed_one("the stock market crashed yesterday")

  def cosine(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x**2 for x in a)) * math.sqrt(sum(x**2 for x in b)))

  sim_related = cosine(vec_a, vec_b)
  sim_unrelated = cosine(vec_a, vec_c)
  assert sim_related > sim_unrelated


def test_singleton_returns_same_instance():
  e1 = get_embedder()
  e2 = get_embedder()
  assert e1 is e2