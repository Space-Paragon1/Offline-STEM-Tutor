from __future__ import annotations
import os
from typing import Optional
from llama_cpp import Llama  # type: ignore

class LocalLLM:
    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = 8):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        # llama.cpp completion API
        out = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</USER_QUESTION>"],
        )
        return out["choices"][0]["text"].strip()
