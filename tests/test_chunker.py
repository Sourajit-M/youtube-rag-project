import sys
import os
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
  sys.path.append(str(root_path))

from app.core.chunker import TranscriptChunker

chunker = TranscriptChunker()

# Simulate a short transcript
sample = '''
Photosynthesis is the process by which plants use sunlight water and carbon dioxide
to produce oxygen and energy in the form of sugar. This process happens in the
chloroplasts of plant cells specifically using a green pigment called chlorophyll.
Chlorophyll absorbs light most efficiently in the red and blue parts of the spectrum
and reflects green light which is why plants appear green to our eyes.
The process of photosynthesis has two main stages. The first stage is called the
light dependent reactions which occur in the thylakoid membranes. During this stage
light energy is captured and used to split water molecules releasing oxygen as a
byproduct. The energy captured is stored in molecules called ATP and NADPH.
The second stage is called the Calvin cycle or light independent reactions which
occur in the stroma of the chloroplast. During this stage the plant uses the ATP
and NADPH produced in the first stage along with carbon dioxide from the air to
build sugar molecules through a series of chemical reactions.
''' * 4   # repeat to get enough tokens to produce multiple chunks

chunks = chunker.chunk(sample)
print(f'Total words: {len(sample.split())}')
print(f'Chunks produced: {len(chunks)}')
print()
for c in chunks:
  word_count = len(c.text.split())
  print(f'  Chunk {c.chunk_index}: tokens {c.start_token}-{c.end_token} ({word_count} words)')
  print(f'    starts: \"{c.text[:60]}...\"')
  print(f'    ends:   \"...{c.text[-60:]}\"')
  print()

# Verify overlap: end of chunk 0 and start of chunk 1 should share words
if len(chunks) >= 2:
  words_0 = chunks[0].text.split()
  words_1 = chunks[1].text.split()
  shared = words_0[-50:]  # last 50 words of chunk 0
  start_1 = words_1[:50]  # first 50 words of chunk 1
  overlap_count = len(set(shared) & set(start_1))
  print(f'Overlap check: ~{overlap_count} shared words between chunk 0 and chunk 1')
  print('(not exact due to repeated words in sample, but should be > 0)')
