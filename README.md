# History AI — FastAPI Backend + Chat Frontend

A local REST API + beautiful chat UI powered by a **Qwen 2.5 (0.5B)** model
fine-tuned on *A History of Ancient and Early Medieval India* by Upinder Singh.

## 🤗 Model on HuggingFace
👉 **[https://huggingface.co/Rut-ai/qwen2.5-upinder-singh-history](https://huggingface.co/Rut-ai/qwen2.5-upinder-singh-history)**

## 🚀 Quick Start
```bash
pip install -r requirements.txt
cd backend
python main.py
# Open frontend/index.html in your browser
```

## API Endpoints
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | Server + model status |
| GET | `/info` | Model info + VRAM |
| POST | `/chat` | Send query → full answer |
| POST | `/chat/stream` | Streaming (SSE) response |
| DELETE | `/chat/history` | Reset conversation |

**Swagger docs:** http://localhost:8000/docs

## Quality Checks
```bash
pytest tests/test_api.py -v   # 35 tests, all passing
```

## Project Structure
```
backend/    FastAPI app + model loader
frontend/   Chat UI (single HTML file)
tests/      35 automated quality checks
```
