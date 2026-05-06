# eval/run_eval.py
"""
Evaluation harness for the RAG pipeline.

Measures two things:
1. Hit rate — did the correct source video appear in the retrieved chunks?
2. Answer quality — does the answer contain expected keywords?

Why this matters for interviews:
"How do you know your RAG system works?"
"I built an eval harness that measures retrieval hit rate and
answer faithfulness against a labelled question set. My system
achieves X% hit rate on N questions."

That answer immediately separates you from candidates who just
built the system and assumed it worked.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.rag import RAGPipeline
from app.core.retriever import HybridRetriever


def run_eval(questions_path: str = "eval/questions.json") -> dict:
  # Load questions
  with open(questions_path) as f:
    questions = json.load(f)

  # Initialise pipeline
  retriever = HybridRetriever()
  retriever.load_or_build_bm25()

  pipeline = RAGPipeline(retriever=retriever)

  results = []
  hit_count = 0
  keyword_count = 0

  print(f"Running eval on {len(questions)} questions...\n")
  print(f"{'ID':<6} {'Hit':<5} {'KW':<5} Question")
  print("-" * 70)

  for q in questions:
    response = pipeline.ask(q["question"], top_k=5)

    # Metric 1: Hit rate
    # Did the expected source video appear in the retrieved sources?
    source_titles = [s["video_title"] for s in response.sources]

    hit = any(
      q["expected_source"].lower() in title.lower()
      or title.lower() in q["expected_source"].lower()
      for title in source_titles
    )

    # Metric 2: Keyword presence
    # Does the answer contain the expected keywords?
    answer_lower = response.answer.lower()

    kw_hit = any(
      kw.lower() in answer_lower
      for kw in q["keywords"]
    )

    if hit:
      hit_count += 1

    if kw_hit:
      keyword_count += 1

    status_hit = "✓" if hit else "✗"
    status_kw = "✓" if kw_hit else "✗"

    print(
      f"{q['id']:<6} "
      f"{status_hit:<5} "
      f"{status_kw:<5} "
      f"{q['question'][:55]}"
    )

    results.append({
      "id": q["id"],
      "question": q["question"],
      "hit": hit,
      "keyword_hit": kw_hit,
      "answer_preview": response.answer[:150],
      "sources_returned": source_titles,
      "expected_source": q["expected_source"],
    })

  total = len(questions)
  hit_rate = hit_count / total
  kw_rate = keyword_count / total

  print("-" * 70)
  print("\nResults:")
  print(f"  Hit rate:      {hit_count}/{total} = {hit_rate:.0%}")
  print(f"  Keyword rate:  {keyword_count}/{total} = {kw_rate:.0%}")
  print()

  if hit_rate >= 0.8:
    print("  Retrieval quality: GOOD (≥80% hit rate)")
  elif hit_rate >= 0.6:
    print(
      "  Retrieval quality: FAIR (60-79%) "
      "— consider tuning chunk_size"
    )
  else:
    print(
      "  Retrieval quality: POOR (<60%) "
      "— check ingestion and embeddings"
    )

  return {
    "hit_rate": hit_rate,
    "keyword_rate": kw_rate,
    "total_questions": total,
    "results": results,
  }


if __name__ == "__main__":
  run_eval()