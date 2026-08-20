# Car price estimator

Interview starter: a FastAPI + LangChain **agent** that estimates a car's market price from a **single-layer Chroma RAG** index. The intended follow-up is to split retrieval into a two-layer pipeline with a **reranker**.

## What you get

- Vanilla HTML frontend (no third-party CSS/JS): brand menu, new/used menu, and a prompt box
- FastAPI API that serves the UI and `/api/estimate`
- LangChain agent with a `search_market_listings` tool
- Chroma vector store seeded from `data/listings.json`

```
Browser (HTML)  →  FastAPI  →  LangChain agent
                                      │
                                      ▼
                               retrieve()  ← single dense search
                                      │
                                      ▼
                                    Chroma
```

## Setup

Python 3.11+ and an OpenAI API key.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Put your key in `.env`:

```
OPENAI_API_KEY=sk-...
```

First request (or app startup) embeds the listings into `./chroma_db`. To rebuild the index:

```powershell
python -m app.rag.ingest --reset
```

Run the server:

```powershell
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

Try a prompt such as: `2020 330i, 42,000 miles, 255 hp, 295 lb-ft, Premium package` with brand **BMW** and condition **Used**.

## Interview task

Retrieval lives in [`app/rag/retriever.py`](app/rag/retriever.py). It is **one ranking pass**: Chroma cosine/embedding similarity.

Change it to a **two-layer** retrieve-then-rerank pipeline:

1. **Recall** — fetch a larger candidate set from Chroma. Raise `RECALL_K` (for example 5 → 20).
2. **Rerank** — implement `rerank(query, documents)` so those candidates are scored against the user query (cross-encoder, LLM-as-judge, or similar) and the best `FINAL_K` are returned.

Keep `retrieve()` as the public entry point so the agent and frontend do not need to change.

The seed listings include near-misses (same brand, similar horsepower/torque, wrong body style) so a reranker should be able to beat pure vector search.

## Project layout

| Path | Role |
| --- | --- |
| `app/main.py` | FastAPI app, static files, API routes |
| `app/agent/estimator.py` | LangChain `create_agent` + RAG tool |
| `app/rag/retriever.py` | Single-layer retriever (**interview focus**) |
| `app/rag/ingest.py` | Load `data/listings.json` into Chroma |
| `app/static/` | HTML / CSS / JS with no CDN libraries |
| `data/listings.json` | Comparable listings used as RAG source |
