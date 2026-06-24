"""
Configuration — loaded from environment variables or .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    # ── Model ──────────────────────────────────────────────────────────────────
    # Use the local merged model by default; fall back to HuggingFace hub ID.
    MODEL_PATH: str = r"C:\Users\comed\.gemini\antigravity-ide\scratch\qlora-finetune\qlora_history\merged_model"
    MODEL_HF_ID: str = "Rut-ai/qwen2.5-upinder-singh-history"  # fallback if local not found
    LOAD_IN_4BIT: bool = True          # use 4-bit QLoRA (saves VRAM)

    # ── Generation defaults ────────────────────────────────────────────────────
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    REPETITION_PENALTY: float = 1.1

    # ── Server ─────────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False

    # ── Safety ────────────────────────────────────────────────────────────────
    MAX_QUERY_LENGTH: int = 1000    # characters
    MAX_HISTORY_TURNS: int = 10     # conversation turns to keep

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
