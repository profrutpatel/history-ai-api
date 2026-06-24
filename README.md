# 🏛️ History AI — API Endpoint

A local API server + chat frontend powered by your **QLoRA fine-tuned Qwen 2.5 model**  
trained on *A History of Ancient and Early Medieval India* by Upinder Singh.

---

## 📁 Project Structure

```
history-api/
├── backend/
│   ├── main.py          ← FastAPI app (all routes)
│   ├── model_loader.py  ← Model loading + inference engine
│   ├── schemas.py       ← Request/response validation
│   └── config.py        ← Settings (edit model path etc.)
├── frontend/
│   └── index.html       ← Chat UI (just open in browser!)
├── tests/
│   └── test_api.py      ← 35 quality checks (pytest)
├── setup.bat            ← One-click install
├── start.bat            ← One-click start server
├── run_tests.bat        ← One-click run tests
└── requirements.txt
```

---

## 🚀 Getting Started — 3 Steps

### Step 1 — Setup (only once)

Double-click **`setup.bat`**  
*(creates a virtual environment and installs all packages)*

### Step 2 — Start the server

Double-click **`start.bat`**

You'll see:
```
API will be live at:  http://localhost:8000
Swagger docs at:      http://localhost:8000/docs
```

Wait ~30 seconds for the model to load (watch for `Model ready. API is live.`)

### Step 3 — Open the chat

Double-click **`frontend/index.html`**  
*(opens in your browser — start chatting!)*

---

## 🌐 API Endpoints

| Method | URL | What it does |
|--------|-----|-------------|
| `GET` | `/health` | Check if model is ready |
| `GET` | `/info` | Model name, device, VRAM usage |
| `POST` | `/chat` | Send a question → get full answer |
| `POST` | `/chat/stream` | Send a question → get streaming answer |
| `DELETE` | `/chat/history` | Reset conversation |

**Interactive docs (Swagger):** http://localhost:8000/docs

---

## 💬 Using the API Directly

### Quick test with PowerShell

```powershell
# Check if API is ready
Invoke-RestMethod http://localhost:8000/health

# Ask a question
$body = @{ query = "Who was Ashoka?" } | ConvertTo-Json
Invoke-RestMethod -Method POST -Uri http://localhost:8000/chat `
  -ContentType "application/json" -Body $body
```

### With Python

```python
import requests

# Simple question
r = requests.post("http://localhost:8000/chat", json={
    "query": "Describe the Indus Valley Civilisation."
})
print(r.json()["answer"])

# Multi-turn conversation
r = requests.post("http://localhost:8000/chat", json={
    "query": "What happened after his death?",
    "history": [
        {"role": "user",      "content": "Who was Chandragupta Maurya?"},
        {"role": "assistant", "content": "Chandragupta Maurya founded the Mauryan Empire in 321 BCE."},
    ]
})
print(r.json()["answer"])
```

### With curl

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about the Gupta Empire."}'
```

### Streaming response (Python)

```python
import requests, json

with requests.post("http://localhost:8000/chat/stream",
                   json={"query": "Explain the Vedic period."},
                   stream=True) as r:
    for line in r.iter_lines():
        if line:
            raw = line.decode().removeprefix("data: ")
            if raw == "[DONE]": break
            token = json.loads(raw).get("token", "")
            print(token, end="", flush=True)
```

---

## ⚙️ Configuration

Edit **`backend/config.py`** (or create a `.env` file next to it):

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL_PATH` | Local merged model path | Path to your fine-tuned model |
| `MODEL_HF_ID` | `Rut-ai/qwen2.5-upinder-singh-history` | HuggingFace fallback |
| `LOAD_IN_4BIT` | `True` | Use 4-bit quantisation (saves VRAM) |
| `MAX_NEW_TOKENS` | `512` | Max tokens per response |
| `TEMPERATURE` | `0.7` | Creativity (0=deterministic, 1=creative) |
| `PORT` | `8000` | Server port |

**.env file example:**
```
MODEL_PATH=C:\path\to\your\merged_model
MAX_NEW_TOKENS=256
TEMPERATURE=0.5
PORT=8080
```

---

## 🧪 Quality Checks

Run all 35 tests:

```
run_tests.bat
```

Or manually:
```
venv\Scripts\activate
cd backend
pytest ..\tests\test_api.py -v
```

**Tests cover:**
- ✅ Health / info endpoints
- ✅ Valid chat requests
- ✅ Conversation history (multi-turn)
- ✅ Input validation (empty, too long, bad fields)
- ✅ Streaming responses
- ✅ Response time benchmarks
- ✅ Swagger/ReDoc docs accessible

---

## 🔄 Use With a Different Model

To swap the fine-tuned model, edit `config.py`:

```python
MODEL_PATH = r"C:\path\to\any\merged_model"
# OR
MODEL_HF_ID = "username/any-hf-model"
```

---

## ❓ Troubleshooting

| Problem | Fix |
|---------|-----|
| `Model not loading` | Wait 30-60s — model loads at startup. Check terminal. |
| `CUDA out of memory` | Set `LOAD_IN_4BIT=True` in config, or reduce `MAX_NEW_TOKENS` |
| `Connection refused` | Make sure `start.bat` is running before opening the chat |
| `Chat UI not connecting` | Check `const API = "http://localhost:8000"` in index.html |
| `Port already in use` | Change `PORT=8001` in config, update `index.html` API URL |
| `Module not found` | Run `setup.bat` again |

---

*Stack: FastAPI · Uvicorn · Pydantic · PyTorch · HuggingFace Transformers · PEFT*
