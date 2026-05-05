import os
import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
  sys.path.append(str(root_path))

from app.core.chunker import TranscriptChunker
from app.core.embedder import get_embedder
from app.db.vectordb import VectorDB
from app.core.retriever import HybridRetriever
from app.core.rag import RAGPipeline

# ── Ingest test data ──────────────────────────────────────────────────────
transcripts = {
  'video_001': {
    'title': 'Photosynthesis Explained',
    'channel': 'CrashCourse',
    'text': '''Photosynthesis is the process by which plants convert
    sunlight into chemical energy stored in glucose. Inside plant cells
    are organelles called chloroplasts which contain a green pigment
    called chlorophyll. Chlorophyll absorbs red and blue light from the
    sun and reflects green light which is why plants appear green.
    The light reactions occur in the thylakoid membranes and produce
    ATP and NADPH. The Calvin cycle uses these energy carriers along
    with carbon dioxide from the air to synthesise glucose molecules.
    Oxygen is released as a byproduct of splitting water molecules
    during the light reactions. This oxygen is what animals breathe.
    Without photosynthesis the oxygen in our atmosphere would not exist
    and complex life on Earth would be impossible.''' * 6,
  },
  'video_002': {
    'title': 'CRISPR Gene Editing',
    'channel': 'CrashCourse',
    'text': '''CRISPR Cas9 is a powerful gene editing technology that
    allows scientists to modify DNA sequences with high precision.
    The system was originally discovered in bacteria as part of their
    immune system to defend against viruses. Jennifer Doudna and
    Emmanuelle Charpentier adapted it into a gene editing tool and
    won the Nobel Prize in Chemistry in 2020 for this work.
    The guide RNA is designed to match the target DNA sequence.
    Cas9 is a protein that acts like molecular scissors cutting the
    DNA at the precise location specified by the guide RNA.
    After the cut the cell tries to repair the break which scientists
    exploit to insert delete or modify specific genes. Applications
    include treating sickle cell disease cancer immunotherapy and
    developing disease resistant crops.''' * 6,
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

# ── Build retriever + RAG pipeline ────────────────────────────────────────
retriever = HybridRetriever()
retriever.rebuild_bm25()
pipeline = RAGPipeline(retriever=retriever)

# ── Ask questions ─────────────────────────────────────────────────────────
questions = [
  'How do plants convert sunlight into energy?',
  'Who won the Nobel Prize for CRISPR?',
  'What is chlorophyll and what does it do?',
]

for question in questions:
  print(f'\nQ: {question}')
  response = pipeline.ask(question)
  print(f'A: {response.answer}')
  print(f'Sources: {[s["video_title"] for s in response.sources]}')
  print(f'Chunks used: {response.chunks_used}')

# ── Cleanup ───────────────────────────────────────────────────────────────
db.delete_video_chunks('video_001')
db.delete_video_chunks('video_002')
print(f'\nCleanup done.')