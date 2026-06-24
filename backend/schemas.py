"""
Pydantic schemas for API request / response validation.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from config import settings


# ── Request models ─────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str

class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=settings.MAX_QUERY_LENGTH,
        description="The user's question or instruction",
    )
    history: list[ChatMessage] = Field(
        default=[],
        max_length=settings.MAX_HISTORY_TURNS * 2,
        description="Previous conversation turns",
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        ge=10,
        le=2048,
        description="Override default max new tokens",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.01,
        le=2.0,
        description="Sampling temperature",
    )
    stream: bool = Field(default=False, description="Enable streaming response")

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query must not be blank or whitespace only")
        return v.strip()


# ── Response models ────────────────────────────────────────────────────────────

class ChatResponse(BaseModel):
    answer: str
    query: str
    model: str
    tokens_generated: int
    generation_time_ms: float

class HealthResponse(BaseModel):
    status: str           # "ok" | "loading" | "error"
    model_loaded: bool
    device: str
    message: str

class InfoResponse(BaseModel):
    model_path: str
    model_loaded: bool
    device: str
    vram_used_gb: Optional[float]
    vram_total_gb: Optional[float]
    max_new_tokens: int
    temperature: float
    top_p: float
    version: str = "1.0.0"

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
