import os
from pathlib import Path
import sys

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
  sys.path.append(str(root_path))

from app.core.embedder import get_embedder

embedder = get_embedder()
print('Model loaded')

# Embed a batch (how ingestion uses it)
chunks = [
    'Photosynthesis converts sunlight into chemical energy.',
    'Chlorophyll absorbs red and blue light, reflects green.',
    'The Calvin cycle produces sugar using carbon dioxide.',
]
vectors = embedder.embed(chunks)
print(f'Batch embed: {len(vectors)} vectors, each {len(vectors[0])} dims')

# Embed a query (how retrieval uses it)
query_vec = embedder.embed_one('how do plants make food')
print(f'Query embed: {len(query_vec)} dims')

# Sanity check: similar texts should have higher cosine similarity
# than unrelated texts — let's verify manually
import math

def cosine(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    mag_a = math.sqrt(sum(x**2 for x in a))
    mag_b = math.sqrt(sum(x**2 for x in b))
    return dot / (mag_a * mag_b)

sim_0_query = cosine(vectors[0], query_vec)
sim_2_query = cosine(vectors[2], query_vec)  # Calvin cycle — less related to 'make food'

print(f'Similarity: chunk0 (photosynthesis) vs query: {sim_0_query:.3f}')
print(f'Similarity: chunk2 (calvin cycle)   vs query: {sim_2_query:.3f}')
print(f'Chunk0 more relevant: {sim_0_query > sim_2_query}  (should be True)')

# Singleton check — same object, model loads once
e2 = get_embedder()
print(f'Singleton works: {embedder is e2}  (should be True)')
