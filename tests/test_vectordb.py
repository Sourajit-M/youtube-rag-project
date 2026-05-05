import sys
import os
from pathlib import Path

# Add project root to sys.path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from app.db.vectordb import VectorDB

db = VectorDB()
print('Collection:', db.stats())

# Upsert some fake chunks
db.upsert_chunks(
    chunks=['Plants use sunlight to make food.', 'This process is called photosynthesis.'],
    embeddings=[[0.1] * 384, [0.2] * 384],   # fake embeddings — real ones come from FastEmbed
    video_youtube_id='test_video_001',
    video_title='Biology 101',
    channel_name='CrashCourse',
)
print('After upsert:', db.stats())

# Upsert same video again — count should stay the same (upsert, not insert)
db.upsert_chunks(
    chunks=['Plants use sunlight to make food.', 'This process is called photosynthesis.'],
    embeddings=[[0.1] * 384, [0.2] * 384],
    video_youtube_id='test_video_001',
    video_title='Biology 101',
    channel_name='CrashCourse',
)
print('After re-upsert (should still be 2):', db.stats())

# Search with a fake query vector
results = db.search(query_embedding=[0.1] * 384, top_k=2)
for r in results:
    print(f'  [{r.similarity:.3f}] chunk {r.chunk_index} — {r.text[:50]}')

# Cleanup test data
db.delete_video_chunks('test_video_001')
print('After cleanup:', db.stats())