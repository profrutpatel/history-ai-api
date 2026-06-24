"""
History AI — FastAPI Backend
==============================
Endpoints:
  GET  /health           — server + model status
  GET  /info             — model metadata and device info
  POST /chat             — send a query, get a full response
  POST /chat/stream      — send a query, get a streaming response (SSE)
  DELETE /chat/history   — clear server-side conversation memory (stateless note)
"""

import sys
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from config import settings
from schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    InfoResponse,
    ErrorResponse,
)
from model_loader import engine

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("history-api")


# ── Lifespan — load model at startup ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info("Starting up — loading model...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, engine.load)
    log.info("Model ready. API is live.")
    yield
    log.info("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="History AI API",
    description=(
        "Chat with a Qwen 2.5 (0.5B) model fine-tuned on "
        "Upinder Singh's *A History of Ancient and Early Medieval India*."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins for local dev (tighten for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request logging middleware ─────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    log.info(f"→ {request.method} {request.url.path}")
    response = await call_next(request)
    log.info(f"← {response.status_code}")
    return response


# ── Error handler ──────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
    log.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
async def health():
    """
    Returns server and model status.
    Use this to check if the API is ready before sending queries.
    """
    if engine.loading_error:
        return HealthResponse(
            status="error",
            model_loaded=False,
            device=engine.device,
            message=f"Model load failed: {engine.loading_error}",
        )
    if not engine.loaded:
        return HealthResponse(
            status="loading",
            model_loaded=False,
            device=engine.device,
            message="Model is still loading, please wait...",
        )
    return HealthResponse(
        status="ok",
        model_loaded=True,
        device=engine.device,
        message="API is ready.",
    )


@app.get(
    "/info",
    response_model=InfoResponse,
    summary="Model information",
    tags=["System"],
)
async def info():
    """Returns model path, device, VRAM usage, and generation settings."""
    vram_used, vram_total = engine.vram_info()
    return InfoResponse(
        model_path=engine.model_id,
        model_loaded=engine.loaded,
        device=engine.device,
        vram_used_gb=vram_used,
        vram_total_gb=vram_total,
        max_new_tokens=settings.MAX_NEW_TOKENS,
        temperature=settings.TEMPERATURE,
        top_p=settings.TOP_P,
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with the model",
    tags=["Chat"],
)
async def chat(request: ChatRequest):
    """
    Send a query and receive a complete answer.

    Optionally pass `history` (list of previous user/assistant turns)
    to maintain conversation context.
    """
    if not engine.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still loading. Please try again in a moment.",
        )

    try:
        loop = asyncio.get_event_loop()
        answer, tokens, time_ms = await loop.run_in_executor(
            None,
            lambda: engine.generate(
                query=request.query,
                history=request.history,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
            ),
        )
    except Exception as e:
        log.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}",
        )

    log.info(
        f"Query: {request.query[:60]}... | "
        f"Tokens: {tokens} | Time: {time_ms:.0f}ms"
    )

    return ChatResponse(
        answer=answer,
        query=request.query,
        model=engine.model_id,
        tokens_generated=tokens,
        generation_time_ms=round(time_ms, 2),
    )


@app.post(
    "/chat/stream",
    summary="Streaming chat (Server-Sent Events)",
    tags=["Chat"],
    response_class=StreamingResponse,
)
async def chat_stream(request: ChatRequest):
    """
    Send a query and receive tokens as a **Server-Sent Events** stream.

    Each SSE event is:
    ```
    data: {"token": "..."}
    ```
    A final `data: [DONE]` signals completion.
    """
    if not engine.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still loading. Please try again in a moment.",
        )

    async def token_generator() -> AsyncIterator[str]:
        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()

        def run_stream():
            try:
                for token in engine.stream_generate(
                    query=request.query,
                    history=request.history,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                ):
                    loop.call_soon_threadsafe(q.put_nowait, token)
            except Exception as e:
                loop.call_soon_threadsafe(q.put_nowait, Exception(str(e)))
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)  # sentinel

        import threading
        t = threading.Thread(target=run_stream)
        t.start()

        while True:
            item = await q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                yield f"data: {json.dumps({'error': str(item)})}\n\n"
                break
            yield f"data: {json.dumps({'token': item})}\n\n"

        yield "data: [DONE]\n\n"
        t.join()

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete(
    "/chat/history",
    summary="Clear conversation history",
    tags=["Chat"],
)
async def clear_history():
    """
    Stateless API note: history is passed per-request.
    This endpoint exists for frontend convenience — it returns a confirmation.
    """
    return {"message": "History cleared. Start a new conversation by omitting the history field."}


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level="info",
    )
