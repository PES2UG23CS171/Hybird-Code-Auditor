from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import requests


class LLMClient(Protocol):
    model_name: str

    def generate(self, prompt: str, *, temperature: float = 0.2) -> str: ...

    def is_available(self) -> bool: ...


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
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(
                f"Ollama request failed for model '{self.model_name}' at {endpoint}: {exc}"
            ) from exc

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url.rstrip('/')}/api/tags", timeout=3)
            return response.ok
        except requests.RequestException:
            return False


@dataclass
class OpenAICompatibleClient:
    """Client for any OpenAI-compatible chat completions endpoint.

    Works with Groq, OpenRouter, Together, a local vLLM server, etc. This is
    what lets the full pipeline run on hosted deployments where Ollama is not
    available.
    """

    base_url: str = "https://api.groq.com/openai/v1"
    api_key: str = ""
    model_name: str = "llama-3.1-8b-instant"

    def generate(self, prompt: str, *, temperature: float = 0.2) -> str:
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            return str(data["choices"][0]["message"]["content"]).strip()
        except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Hosted LLM request failed for model '{self.model_name}' at {endpoint}: {exc}"
            ) from exc

    def is_available(self) -> bool:
        return bool(self.api_key)
