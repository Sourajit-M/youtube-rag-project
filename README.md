# YouTube RAG Engine

A production-grade Retrieval-Augmented Generation (RAG) system that lets
you search and ask questions across any YouTube channel's transcript content.

**Live demo:** [your-deployment-url]  
**API docs:** [your-deployment-url/docs]

---

## What it does

- Add any YouTube channel → transcripts are automatically extracted,
  chunked, and embedded
- Ask natural language questions → answers grounded in real video content,
  with source citations
- Hybrid retrieval (BM25 + vector search) ensures both exact term matching
  and semantic similarity
- Provider-agnostic LLM layer via LiteLLM — swap Groq, Gemini, or any
  provider with one environment variable

## Architecture
User query
│
▼
FastAPI (/ask, /search, /channels, /health)
│
├── Hybrid Retriever
│       ├── BM25 (rank-bm25) — exact term matching
│       ├── Vector search (ChromaDB) — semantic similarity
│       └── RRF fusion — combines both ranked lists
│
├── LiteLLM router — Groq (primary) → Gemini (fallback)
│
└── RAG pipeline — retrieved chunks → prompt → grounded answer
Background scheduler (APScheduler) polls channels hourly for new videos.
Ingestion: YouTube API → yt-dlp → TranscriptChunker → FastEmbed → ChromaDB

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| API | FastAPI + Pydantic v2 | Async, auto-docs, type-safe |
| LLM | LiteLLM (Groq + Gemini) | Provider-agnostic, one-line switching |
| Vector DB | ChromaDB | Persistent, local-first, HNSW indexing |
| Keyword search | rank-bm25 | Exact term recall, same algo as Elasticsearch |
| Fusion | Reciprocal Rank Fusion | No score normalisation needed |
| Embeddings | FastEmbed (MiniLM-L6-v2) | ONNX runtime, CPU-only, 384-dim |
| Metadata DB | SQLite + SQLModel | Channels, videos, job tracking |
| Scheduler | APScheduler | In-process, no Redis needed |
| Packaging | uv | 10-100x faster than pip, lock files |
| Container | Docker (multi-stage) | Slim runtime image |

## Eval results

Evaluated on 5 labelled questions against 3 ingested CrashCourse videos:

| Metric | Score |
|---|---|
| Retrieval hit rate | 5/5 (100%) |
| Answer keyword rate | 5/5 (100%) |

## Quickstart

### Prerequisites
- Python 3.11+
- uv (`pip install uv`)
- API keys: Groq, Gemini, YouTube Data API v3

### Setup

```bash
git clone https://github.com/yourname/youtube-rag-engine
cd youtube-rag-engine

uv sync

cp .env.example .env
# Edit .env with your API keys

uv run uvicorn app.api.main:app --reload --port 8000
```

### Add a channel and ask questions

```bash
# Add a YouTube channel
curl -X POST http://localhost:8000/channels \
  -H "Content-Type: application/json" \
  -d '{"channel_input": "@crashcourse", "max_videos": 10}'

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How was the Earth formed?"}'

# Search without LLM generation
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "plate tectonics"}'
```

### Run tests

```bash
uv run pytest tests/ -v
# 27 passed
```

### Run eval

```bash
uv run python eval/run_eval.py
```

### Docker

```bash
docker build -t youtube-rag-engine .
docker run -p 8000:8000 --env-file .env youtube-rag-engine
```

## Project structure
app/
├── api/          # FastAPI routes, Pydantic models
├── core/         # Chunker, embedder, retriever, LLM, RAG pipeline
├── db/           # ChromaDB handler, SQLite models
└── ingestion/    # YouTube API, transcript extraction, pipeline, scheduler
eval/             # Labelled questions + evaluation harness
tests/            # 27 pytest tests covering all core modules


## Key design decisions

**Why chunking over full-transcript embedding**  
Embedding entire transcripts dilutes meaning across topics. 300-token
chunks with 50-token overlap keep each embedding focused on one idea,
improving retrieval precision significantly.

**Why hybrid search over pure vector search**  
Vector search misses exact terms (a query for "CRISPR" may not retrieve
chunks that don't discuss gene editing semantically). BM25 catches exact
terms but misses meaning. RRF fusion covers both blind spots.

**Why LiteLLM**  
Provider lock-in is a real production risk. LiteLLM gives a single
`completion()` interface over 100+ providers. Switching from Groq to
Gemini requires changing one environment variable, not refactoring code.

**Why APScheduler over Celery**  
Celery requires a Redis broker — unnecessary operational complexity for
hourly polling at this scale. APScheduler runs in-process. The upgrade
path to Celery is clear if throughput demands it.