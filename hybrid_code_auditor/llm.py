from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class OllamaClient:
    base_url: str = "http://localhost:11434"
    model_name: str = "llama3.2"

    def generate(self, prompt: str, *, temperature: float = 0.2) -> str:
        endpoint = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        try:
            response = requests.post(endpoint, json=payload, timeout=300)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            return str(data.get("response", "")).strip()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Ollama request failed for model '{self.model_name}' at {endpoint}: {exc}"
            ) from exc

