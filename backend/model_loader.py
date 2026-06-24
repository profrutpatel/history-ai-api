"""
Model loader and inference engine.

Loads the fine-tuned Qwen 2.5 model (local merged model or HuggingFace hub)
and provides generate() and stream_generate() methods used by the API.
"""

import os
import sys
import time
import asyncio
import threading
import queue
from pathlib import Path
from typing import Iterator, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)

# Force UTF-8 on Windows
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import settings


SYSTEM_PROMPT = (
    "You are a knowledgeable assistant specialised in ancient and early "
    "medieval Indian history, trained on Upinder Singh's scholarship. "
    "Answer questions clearly and accurately based on your training."
)


class ModelEngine:
    """Singleton model engine — loaded once at startup."""

    _instance: Optional["ModelEngine"] = None

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.model_id = ""
        self.loaded = False
        self.loading_error: Optional[str] = None
        self._lock = threading.Lock()

    # ── Singleton ──────────────────────────────────────────────────────────────

    @classmethod
    def get(cls) -> "ModelEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Device Detection ───────────────────────────────────────────────────────

    def _detect_device(self) -> str:
        if torch.cuda.is_available():
            gpu  = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"[GPU] {gpu}  ({vram:.1f} GB VRAM)")
            return "cuda"
        print("[WARN] No CUDA GPU — running on CPU (slow)")
        return "cpu"

    # ── Model Loading ──────────────────────────────────────────────────────────

    def load(self):
        """Load model and tokenizer. Called once at startup."""
        print("\n" + "=" * 55)
        print("  History AI — Loading Fine-tuned Model")
        print("=" * 55)

        self.device = self._detect_device()

        # Resolve model source: prefer local, fall back to HF hub
        local_path = Path(settings.MODEL_PATH)
        if local_path.exists():
            self.model_id = str(local_path)
            print(f"[MODEL] Local model: {self.model_id}")
        else:
            self.model_id = settings.MODEL_HF_ID
            print(f"[MODEL] HuggingFace: {self.model_id}")

        try:
            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                padding_side="left",
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token    = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Optional 4-bit quantisation
            bnb_config = None
            if self.device == "cuda" and settings.LOAD_IN_4BIT:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                print("[MODEL] 4-bit NF4 quantisation enabled")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto"    if self.device == "cuda" else None,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                attn_implementation="eager",
            )
            if self.device != "cuda":
                self.model = self.model.to(self.device)

            self.model.eval()
            params = sum(p.numel() for p in self.model.parameters()) / 1e6
            print(f"[MODEL] Loaded  ({params:.1f}M params)  device={self.device}")
            print("=" * 55 + "\n")

            self.loaded = True

        except Exception as e:
            self.loading_error = str(e)
            print(f"[ERROR] Model load failed: {e}")
            raise

    # ── Prompt Building ────────────────────────────────────────────────────────

    def _build_messages(self, query: str, history: list) -> list[dict]:
        """Build ChatML messages list from query + history."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in history:
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": query})
        return messages

    def _apply_template(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # ── Inference ──────────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        history: list = [],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> tuple[str, int, float]:
        """
        Generate a response synchronously.

        Returns:
            (answer_text, tokens_generated, time_ms)
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded yet.")

        max_tok  = max_new_tokens or settings.MAX_NEW_TOKENS
        temp     = temperature    or settings.TEMPERATURE

        messages = self._build_messages(query, history)
        text     = self._apply_template(messages)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with self._lock, torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tok,
                temperature=temp,
                top_p=settings.TOP_P,
                do_sample=(temp > 0),
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=settings.REPETITION_PENALTY,
            )
        elapsed_ms = (time.time() - t0) * 1000

        generated    = output_ids[0][input_len:]
        answer       = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        tokens_count = len(generated)

        return answer, tokens_count, elapsed_ms

    def stream_generate(
        self,
        query: str,
        history: list = [],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """
        Yield tokens one by one for streaming responses.
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded yet.")

        max_tok = max_new_tokens or settings.MAX_NEW_TOKENS
        temp    = temperature    or settings.TEMPERATURE

        messages = self._build_messages(query, history)
        text     = self._apply_template(messages)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tok,
            temperature=temp,
            top_p=settings.TOP_P,
            do_sample=(temp > 0),
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=settings.REPETITION_PENALTY,
            streamer=streamer,
        )

        # Run generation in background thread so we can yield from main thread
        thread = threading.Thread(
            target=self.model.generate,
            kwargs=gen_kwargs,
        )
        thread.start()

        for token_text in streamer:
            yield token_text

        thread.join()

    # ── System Info ────────────────────────────────────────────────────────────

    def vram_info(self) -> tuple[Optional[float], Optional[float]]:
        """Return (used_gb, total_gb) for the current GPU, or (None, None)."""
        if self.device != "cuda":
            return None, None
        used  = torch.cuda.memory_allocated(0) / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return round(used, 2), round(total, 2)


# Module-level singleton
engine = ModelEngine.get()
