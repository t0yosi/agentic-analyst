"""Request and response models for the HTTP API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """A question for the analyst."""

    question: str = Field(..., min_length=1, max_length=2000)
    remember: bool = Field(default=True, description="Keep this turn in conversation memory.")


class AskResponse(BaseModel):
    """The analyst's answer and what it took to produce it."""

    answer: str
    chart_base64: Optional[str] = None
    agents_called: List[str] = Field(default_factory=list)
    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class HealthResponse(BaseModel):
    """Service liveness and readiness."""

    status: str
    model: str
    database_ready: bool
    tables: dict
