"""
API Quality Checks — pytest test suite
========================================
Tests all endpoints for correctness, validation, and robustness.

Uses module-level mocking of torch/transformers so tests run instantly
on any machine — no GPU or heavy ML packages required in the test env.

Run with:
    pytest tests/test_api.py -v
"""

import sys
import os
import time
import types
import pytest
from unittest.mock import MagicMock, patch

# ── Patch heavy ML imports BEFORE anything from backend is imported ────────────

def _make_mock_module(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod

# torch (and sub-modules used by model_loader / main)
torch_mock = _make_mock_module("torch")
torch_mock.cuda = MagicMock()
torch_mock.cuda.is_available = MagicMock(return_value=True)
torch_mock.cuda.get_device_name = MagicMock(return_value="Mock GPU")
torch_mock.cuda.get_device_properties = MagicMock(
    return_value=MagicMock(total_memory=4 * 1024**3)
)
torch_mock.cuda.memory_allocated = MagicMock(return_value=2.1 * 1024**3)
torch_mock.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False)))
torch_mock.bfloat16 = "bfloat16"
torch_mock.float32  = "float32"
torch_mock.float16  = "float16"
torch_mock.backends = MagicMock()
torch_mock.backends.mps = MagicMock()
torch_mock.backends.mps.is_available = MagicMock(return_value=False)

for sub in ["nn", "cuda.amp", "utils", "distributed"]:
    _make_mock_module(f"torch.{sub}")

# transformers
tf_mock = _make_mock_module("transformers")
tf_mock.AutoModelForCausalLM = MagicMock()
tf_mock.AutoTokenizer = MagicMock()
tf_mock.BitsAndBytesConfig = MagicMock()
tf_mock.TextIteratorStreamer = MagicMock()

# peft / bitsandbytes / accelerate
for pkg in ["peft", "bitsandbytes", "accelerate", "sentencepiece"]:
    _make_mock_module(pkg)

# ── Now add backend to path and import app ─────────────────────────────────────

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

# ── Mock answer ────────────────────────────────────────────────────────────────

MOCK_ANSWER = (
    "The Mauryan Empire, founded by Chandragupta Maurya in c. 321 BCE, "
    "was one of the largest empires of ancient India. Ashoka, its most "
    "famous ruler, embraced Buddhism after the Kalinga War and promoted "
    "Dhamma across the subcontinent."
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Create FastAPI test client with fully mocked model engine."""
    from fastapi.testclient import TestClient
    from contextlib import asynccontextmanager
    from typing import AsyncIterator

    # Build mock engine
    mock_engine = MagicMock()
    mock_engine.loaded = True
    mock_engine.loading_error = None
    mock_engine.device = "cuda"
    mock_engine.model_id = "Rut-ai/qwen2.5-upinder-singh-history"
    mock_engine.generate.return_value = (MOCK_ANSWER, 87, 1240.5)
    mock_engine.vram_info.return_value = (2.1, 4.0)

    def fake_stream(*args, **kwargs):
        for word in MOCK_ANSWER.split():
            yield word + " "

    mock_engine.stream_generate.side_effect = fake_stream

    # Patch lifespan to skip model.load()
    @asynccontextmanager
    async def mock_lifespan(app) -> AsyncIterator[None]:
        yield

    with patch("main.lifespan", mock_lifespan), \
         patch("main.engine",   mock_engine):
        import importlib
        import main as main_mod
        importlib.reload(main_mod)
        main_mod.app.router.lifespan_context = mock_lifespan

        with patch("main.engine", mock_engine):
            yield TestClient(main_mod.app)


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────────────────────────────────────

class TestHealth:

    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"

    def test_health_status_ok_when_model_loaded(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "ok"
        assert r.json()["model_loaded"] is True

    def test_health_has_required_fields(self, client):
        body = client.get("/health").json()
        for f in ("status", "model_loaded", "device", "message"):
            assert f in body, f"Missing field: {f}"

    def test_health_response_under_500ms(self, client):
        t0 = time.time()
        client.get("/health")
        assert (time.time() - t0) * 1000 < 500


# ─────────────────────────────────────────────────────────────────────────────
# INFO
# ─────────────────────────────────────────────────────────────────────────────

class TestInfo:

    def test_info_returns_200(self, client):
        assert client.get("/info").status_code == 200

    def test_info_has_model_path(self, client):
        body = client.get("/info").json()
        assert len(body["model_path"]) > 0

    def test_info_has_valid_device(self, client):
        body = client.get("/info").json()
        assert body["device"] in ("cuda", "mps", "cpu")

    def test_info_has_generation_defaults(self, client):
        body = client.get("/info").json()
        assert isinstance(body["max_new_tokens"], int)
        assert 0 < body["temperature"] <= 2.0
        assert 0 < body["top_p"] <= 1.0

    def test_info_has_vram_fields(self, client):
        body = client.get("/info").json()
        assert "vram_used_gb" in body
        assert "vram_total_gb" in body


# ─────────────────────────────────────────────────────────────────────────────
# CHAT — VALID REQUESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestChatValid:

    def test_chat_200_for_valid_query(self, client):
        r = client.post("/chat", json={"query": "Who was Ashoka?"})
        assert r.status_code == 200, r.text

    def test_chat_response_has_answer(self, client):
        r = client.post("/chat", json={"query": "Describe the Mauryan Empire."})
        assert len(r.json()["answer"]) > 0

    def test_chat_has_all_fields(self, client):
        r = client.post("/chat", json={"query": "Tell me about ancient India."})
        for f in ("answer", "query", "model", "tokens_generated", "generation_time_ms"):
            assert f in r.json(), f"Missing: {f}"

    def test_chat_echoes_query(self, client):
        q = "What is the Indus Valley Civilisation?"
        assert client.post("/chat", json={"query": q}).json()["query"] == q

    def test_chat_tokens_positive(self, client):
        assert client.post("/chat", json={"query": "Gupta period?"}).json()["tokens_generated"] > 0

    def test_chat_time_positive(self, client):
        assert client.post("/chat", json={"query": "What is Dhamma?"}).json()["generation_time_ms"] > 0

    def test_chat_with_history(self, client):
        r = client.post("/chat", json={
            "query": "What happened after his reign?",
            "history": [
                {"role": "user",      "content": "Who was Chandragupta Maurya?"},
                {"role": "assistant", "content": "He founded the Mauryan Empire in 321 BCE."},
            ],
        })
        assert r.status_code == 200
        assert len(r.json()["answer"]) > 0

    def test_chat_custom_max_tokens(self, client):
        r = client.post("/chat", json={"query": "Brief answer: Who was Ashoka?", "max_new_tokens": 50})
        assert r.status_code == 200

    def test_chat_custom_temperature(self, client):
        r = client.post("/chat", json={"query": "Tell me about ancient India.", "temperature": 0.3})
        assert r.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# CHAT — VALIDATION & ERRORS
# ─────────────────────────────────────────────────────────────────────────────

class TestChatValidation:

    def test_empty_query_422(self, client):
        assert client.post("/chat", json={"query": ""}).status_code == 422

    def test_whitespace_query_422(self, client):
        assert client.post("/chat", json={"query": "   "}).status_code == 422

    def test_missing_query_422(self, client):
        assert client.post("/chat", json={}).status_code == 422

    def test_query_too_long_422(self, client):
        assert client.post("/chat", json={"query": "a" * 1001}).status_code == 422

    def test_temperature_too_high_422(self, client):
        assert client.post("/chat", json={"query": "test", "temperature": 5.0}).status_code == 422

    def test_max_tokens_too_high_422(self, client):
        assert client.post("/chat", json={"query": "test", "max_new_tokens": 99999}).status_code == 422

    def test_invalid_history_role_422(self, client):
        r = client.post("/chat", json={
            "query": "test",
            "history": [{"role": "banana", "content": "hello"}],
        })
        assert r.status_code == 422

    def test_get_chat_not_allowed(self, client):
        assert client.get("/chat").status_code == 405


# ─────────────────────────────────────────────────────────────────────────────
# STREAMING
# ─────────────────────────────────────────────────────────────────────────────

class TestChatStream:

    def test_stream_returns_200(self, client):
        r = client.post("/chat/stream", json={"query": "Who was Ashoka?"})
        assert r.status_code == 200

    def test_stream_content_type_event_stream(self, client):
        r = client.post("/chat/stream", json={"query": "Who was Ashoka?"})
        assert "text/event-stream" in r.headers.get("content-type", "")

    def test_stream_ends_with_done(self, client):
        r = client.post("/chat/stream", json={"query": "Who was Ashoka?"})
        assert "[DONE]" in r.text

    def test_stream_empty_query_422(self, client):
        assert client.post("/chat/stream", json={"query": ""}).status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# CLEAR HISTORY
# ─────────────────────────────────────────────────────────────────────────────

class TestClearHistory:

    def test_clear_returns_200(self, client):
        assert client.delete("/chat/history").status_code == 200

    def test_clear_has_message(self, client):
        assert "message" in client.delete("/chat/history").json()


# ─────────────────────────────────────────────────────────────────────────────
# DOCS
# ─────────────────────────────────────────────────────────────────────────────

class TestDocs:

    def test_swagger_accessible(self, client):
        assert client.get("/docs").status_code == 200

    def test_redoc_accessible(self, client):
        assert client.get("/redoc").status_code == 200

    def test_openapi_json_accessible(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        assert r.json()["info"]["title"] == "History AI API"
