from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent.estimator import estimate_price
from app.config import settings
from app.rag.ingest import ensure_vectorstore, load_listings
from app.schemas import EstimateRequest, EstimateResponse, OptionsResponse

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if settings.openai_api_key:
        ensure_vectorstore()
    yield


app = FastAPI(
    title="Car Price Estimator",
    description="LangChain agent over a single-layer Chroma RAG index of car listings.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "ok": True,
        "openai_key_configured": bool(settings.openai_api_key),
    }


@app.get("/api/options", response_model=OptionsResponse)
def options() -> OptionsResponse:
    listings = load_listings()
    brands = sorted({item["brand"] for item in listings})
    return OptionsResponse(brands=brands, conditions=["new", "used"])


@app.post("/api/estimate", response_model=EstimateResponse)
def estimate(payload: EstimateRequest) -> EstimateResponse:
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is missing. Copy .env.example to .env and add a key.",
        )
    try:
        return estimate_price(payload.brand, payload.condition, payload.prompt)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
