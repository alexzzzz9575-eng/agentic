from typing import Literal

from pydantic import BaseModel, Field


class EstimateRequest(BaseModel):
    brand: str = Field(..., min_length=1)
    condition: Literal["new", "used"]
    prompt: str = Field(..., min_length=1, max_length=2000)


class OptionsResponse(BaseModel):
    brands: list[str]
    conditions: list[Literal["new", "used"]]


class PriceEstimate(BaseModel):
    """Structured estimate returned by the LangChain agent."""

    low: int = Field(..., description="Conservative low USD price")
    mid: int = Field(..., description="Most likely USD price")
    high: int = Field(..., description="Optimistic high USD price")
    currency: str = Field(default="USD")
    confidence: Literal["low", "medium", "high"]
    rationale: str = Field(..., description="Short explanation citing comparable listings")


class ListingHit(BaseModel):
    id: str
    brand: str
    model: str
    year: int
    condition: str
    price: int
    horsepower: int | None = None
    torque_lbft: int | None = None
    mileage: int | None = None
    summary: str


class EstimateResponse(BaseModel):
    estimate: PriceEstimate
    comparables: list[ListingHit]
