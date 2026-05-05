import sys
import os
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
  sys.path.append(str(root_path))

from app.core.chunker import TranscriptChunker
from app.core.embedder import get_embedder
from app.db.vectordb import VectorDB
from app.core.retriever import HybridRetriever

# ── Step 1: Ingest some test data ──────────────────────────────────────────
transcripts = {
  'video_001': {
    'title': 'Photosynthesis Explained',
    'channel': 'CrashCourse',
    'text': '''Photosynthesis is the process by which plants convert sunlight
        into chemical energy. Inside the chloroplasts, chlorophyll absorbs light
        and uses it to split water molecules. This releases oxygen into the air.
        The energy captured drives the Calvin cycle which fixes carbon dioxide
        into glucose. Without photosynthesis almost no life on Earth would exist.
        Plants are the foundation of every food chain on the planet.''' * 8,
  },
  'video_002': {
    'title': 'CRISPR Gene Editing',
    'channel': 'CrashCourse',
    'text': '''CRISPR Cas9 is a revolutionary gene editing tool derived from
        bacterial immune systems. Scientists use it to cut DNA at precise locations.
        The guide RNA directs the Cas9 protein to the exact gene sequence.
        Once the DNA is cut the cell attempts to repair itself. Researchers can
        insert delete or modify genes with unprecedented precision. CRISPR has
        potential applications in treating genetic diseases cancer and agriculture.
        The technology won the Nobel Prize in Chemistry in 2020.''' * 8,
  },
}

chunker = TranscriptChunker()
embedder = get_embedder()
db = VectorDB()

for video_id, data in transcripts.items():
  chunks = chunker.chunk_texts(data['text'])
  embeddings = embedder.embed(chunks)
  db.upsert_chunks(
    chunks=chunks,
    embeddings=embeddings,
    video_youtube_id=video_id,
    video_title=data['title'],
    channel_name=data['channel'],
  )
  print(f'Ingested {video_id}: {len(chunks)} chunks')

# ── Step 2: Build hybrid retriever ────────────────────────────────────────
retriever = HybridRetriever()
retriever.rebuild_bm25()

# ── Step 3: Search and inspect ────────────────────────────────────────────
queries = [
  'how do plants produce energy',        # should hit video_001
  'what is CRISPR and how does it work', # should hit video_002
  'Nobel Prize gene technology',         # exact term + semantic
]

for query in queries:
  print(f'\nQuery: "{query}"')
  results = retriever.search(query, top_k=3)
  for r in results:
    print(f'  [{r["rrf_score"]:.5f}] {r["video_title"]} — chunk {r["chunk_index"]}')
    print(f'           "{r["text"][:80]}..."')

# ── Cleanup ───────────────────────────────────────────────────────────────
db.delete_video_chunks('video_001')
db.delete_video_chunks('video_002')
print(f'\nCleanup done. Chunks remaining: {db.count()}')