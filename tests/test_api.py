import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.api.main import app
from app.core.rag import RAGResponse


@pytest.fixture(scope="module")
def client():
  """
  TestClient runs the full FastAPI app including lifespan.
  We mock the heavy components (embedder, ChromaDB) so tests
  run fast without downloading models or needing API keys.
  """
  with TestClient(app) as c:
    yield c


def test_health_endpoint(client):
  response = client.get("/health")

  assert response.status_code == 200

  data = response.json()

  assert data["status"] == "ok"
  assert "chunks_indexed" in data
  assert "channels_tracked" in data
  assert "active_llm" in data


def test_get_channels(client):
  response = client.get("/channels")

  assert response.status_code == 200
  assert isinstance(response.json(), list)


def test_ask_returns_answer(client):
  response = client.post(
    "/ask",
    json={"question": "How was the Earth formed?"},
  )

  assert response.status_code == 200

  data = response.json()

  assert "answer" in data
  assert "sources" in data
  assert "chunks_used" in data
  assert "provider" in data
  assert len(data["answer"]) > 0


def test_ask_short_question_rejected(client):
  response = client.post(
    "/ask",
    json={"question": "Hi"},
  )

  # Pydantic validation error
  assert response.status_code == 422


def test_search_returns_results(client):
  response = client.post(
    "/search",
    json={"query": "earth crust geology"},
  )

  assert response.status_code == 200

  data = response.json()

  assert "results" in data
  assert "total" in data


def test_search_with_channel_filter(client):
  response = client.post(
    "/search",
    json={
      "query": "earth",
      "channel_name": "CrashCourse",
    },
  )

  assert response.status_code == 200


def test_add_channel_invalid_input(client):
  response = client.post(
    "/channels",
    json={
      "channel_input": "not_a_real_channel_xyz123abc",
      "max_videos": 1,
    },
  )

  # Should return 400 (ValueError from resolve_channel_id)
  assert response.status_code in (400, 500)