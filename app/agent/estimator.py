"""LangChain agent that estimates price from RAG comparables."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.config import settings
from app.rag.retriever import retrieve
from app.schemas import EstimateResponse, ListingHit, PriceEstimate

SYSTEM_PROMPT = """You are a used-and-new car price analyst.

Always call search_market_listings before you produce an estimate.
Base the dollar range on the retrieved comparables, not on generic knowledge.
If the comparables are a poor match (wrong body style, much higher horsepower,
or new vs used mismatch), lower your confidence and widen the range.

Return:
- low / mid / high prices in USD integers
- confidence: low, medium, or high
- a short rationale that cites specific retrieved listings
"""


def _format_docs(documents: list[Document]) -> str:
    if not documents:
        return "No comparable listings were found."

    blocks: list[str] = []
    for index, doc in enumerate(documents, start=1):
        meta = doc.metadata
        header = (
            f"[{index}] {meta.get('year')} {meta.get('brand')} {meta.get('model')} "
            f"| {meta.get('condition')} | ${int(meta.get('price', 0)):,}"
        )
        specs = (
            f"Mileage {int(meta.get('mileage', 0)):,} | "
            f"{meta.get('horsepower')} hp | {meta.get('torque_lbft')} lb-ft"
        )
        blocks.append(f"{header}\n{specs}\n{doc.page_content}")
    return "\n\n".join(blocks)


def _to_hits(documents: list[Document]) -> list[ListingHit]:
    hits: list[ListingHit] = []
    for doc in documents:
        meta = doc.metadata
        hits.append(
            ListingHit(
                id=str(meta.get("id", "")),
                brand=str(meta.get("brand", "")),
                model=str(meta.get("model", "")),
                year=int(meta.get("year", 0)),
                condition=str(meta.get("condition", "")),
                price=int(meta.get("price", 0)),
                horsepower=meta.get("horsepower"),
                torque_lbft=meta.get("torque_lbft"),
                mileage=meta.get("mileage"),
                summary=doc.page_content,
            )
        )
    return hits


def estimate_price(brand: str, condition: str, prompt: str) -> EstimateResponse:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    hits: list[Document] = []

    def search_market_listings(query: str) -> str:
        """Search comparable vehicle listings in the local market database.

        Pass a natural-language query that includes model, year, mileage,
        horsepower, torque, and trim when the customer provided them.
        """
        found = retrieve(query, brand=brand, condition=condition)
        hits.clear()
        hits.extend(found)
        return _format_docs(found)

    model = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )
    agent = create_agent(
        model=model,
        tools=[search_market_listings],
        system_prompt=SYSTEM_PROMPT,
        response_format=PriceEstimate,
    )

    user_message = (
        f"Brand: {brand}\n"
        f"Condition: {condition}\n"
        f"Customer description:\n{prompt.strip()}"
    )
    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]})
    structured = result.get("structured_response")
    if not isinstance(structured, PriceEstimate):
        raise RuntimeError("The agent did not return a structured price estimate.")

    return EstimateResponse(estimate=structured, comparables=_to_hits(hits))
