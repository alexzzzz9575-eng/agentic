"""Load car listings into the Chroma collection."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from langchain_core.documents import Document

from app.config import settings
from app.rag.vectorstore import get_vectorstore


def listing_to_text(item: dict) -> str:
    mileage = item.get("mileage", 0)
    return (
        f"{item['year']} {item['brand']} {item['model']} ({item['condition']}). "
        f"Mileage: {mileage:,} miles. "
        f"Horsepower: {item['horsepower']} hp. "
        f"Torque: {item['torque_lbft']} lb-ft. "
        f"Drivetrain: {item['drivetrain']}. "
        f"Listed price: ${item['price']:,} USD. "
        f"Location: {item['location']}. "
        f"{item['notes']}"
    )


def load_listings() -> list[dict]:
    return json.loads(settings.data_path.read_text(encoding="utf-8"))


def listings_to_documents(listings: list[dict] | None = None) -> list[Document]:
    listings = listings if listings is not None else load_listings()
    documents: list[Document] = []
    for item in listings:
        documents.append(
            Document(
                page_content=listing_to_text(item),
                metadata={
                    "id": item["id"],
                    "brand": item["brand"],
                    "model": item["model"],
                    "year": int(item["year"]),
                    "condition": item["condition"],
                    "price": int(item["price"]),
                    "horsepower": int(item["horsepower"]),
                    "torque_lbft": int(item["torque_lbft"]),
                    "mileage": int(item["mileage"]),
                },
            )
        )
    return documents


def _collection_size(vectorstore) -> int:
    snapshot = vectorstore.get()
    ids = snapshot.get("ids") if snapshot else None
    return len(ids) if ids else 0


def ensure_vectorstore(*, reset: bool = False) -> int:
    """Create the Chroma collection if it is empty. Returns document count."""
    if reset and settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    vectorstore = get_vectorstore()
    existing = _collection_size(vectorstore)
    if existing > 0:
        return existing

    documents = listings_to_documents()
    ids = [str(doc.metadata["id"]) for doc in documents]
    vectorstore.add_documents(documents, ids=ids)
    return len(documents)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest car listings into Chroma.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing Chroma directory and re-ingest.",
    )
    args = parser.parse_args()
    count = ensure_vectorstore(reset=args.reset)
    print(f"Chroma collection '{settings.collection_name}' holds {count} listings.")
    print(f"Persist directory: {Path(settings.chroma_dir).resolve()}")
