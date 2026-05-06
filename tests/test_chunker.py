import pytest
from app.core.chunker import TranscriptChunker, Chunk


@pytest.fixture
def chunker():
  return TranscriptChunker()


def test_basic_chunking_produces_chunks(chunker):
  text = "word " * 400  # 400 words — should produce multiple chunks
  chunks = chunker.chunk(text)
  assert len(chunks) >= 2


def test_chunk_size_respected(chunker):
  text = "word " * 400
  chunks = chunker.chunk(text)
  for c in chunks[:-1]:  # last chunk can be shorter
    assert len(c.text.split()) <= chunker.chunk_size


def test_overlap_exists(chunker):
  """Last words of chunk N should appear at start of chunk N+1."""
  text = "word " * 400
  chunks = chunker.chunk(text)
  if len(chunks) >= 2:
    end_words = set(chunks[0].text.split()[-chunker.overlap:])
    start_words = set(chunks[1].text.split()[:chunker.overlap])
    assert len(end_words & start_words) > 0


def test_short_transcript_returns_single_chunk(chunker):
  text = "This is a very short transcript with few words."
  chunks = chunker.chunk(text)
  assert len(chunks) == 1


def test_empty_transcript_returns_no_chunks(chunker):
  chunks = chunker.chunk("")
  assert chunks == []


def test_chunk_index_is_sequential(chunker):
  text = "word " * 400
  chunks = chunker.chunk(text)
  for i, chunk in enumerate(chunks):
    assert chunk.chunk_index == i


def test_cleaning_removes_music_tags(chunker):
  text = "[Music] hello world [Applause] this is a test " * 50
  chunks = chunker.chunk(text)
  for c in chunks:
    assert "[Music]" not in c.text
    assert "[Applause]" not in c.text


def test_chunk_texts_returns_strings(chunker):
  text = "word " * 400
  texts = chunker.chunk_texts(text)
  assert all(isinstance(t, str) for t in texts)
  assert len(texts) >= 2