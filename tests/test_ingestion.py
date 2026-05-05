import os
import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
  sys.path.append(str(root_path))

from app.db.sqlite import create_tables
from app.ingestion.pipeline import IngestionPipeline
from app.core.retriever import HybridRetriever
from app.core.rag import RAGPipeline

create_tables()
pipeline = IngestionPipeline()

# Test with a single real video first
# CrashCourse Biology: The Chemistry of Life (~10 min, has transcripts)
result = pipeline.add_channel('@crashcourse', max_videos=3)
print(result)

# Now ask a real question
retriever = HybridRetriever()
retriever.load_or_build_bm25()
rag = RAGPipeline(retriever=retriever)

response = rag.ask('What topics does CrashCourse cover?')
print()
print('Answer:', response.answer[:300])
print('Sources:', [s['video_title'] for s in response.sources])
