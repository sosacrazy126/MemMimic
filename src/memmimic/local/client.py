"""
Ollama client integration for Clay-Bridge.
"""

import logging
from dataclasses import dataclass
from typing import List

import requests

logger = logging.getLogger(__name__)


@dataclass
class OllamaResponse:
    """Ollama API response."""

    response: str
    model: str
    done: bool
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0
    eval_count: int = 0


class OllamaClient:
    """Simple Ollama API client."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = requests.Session()

    def generate(self, model: str, prompt: str, **kwargs) -> OllamaResponse:
        """Generate response from Ollama model."""

        payload = {"model": model, "prompt": prompt, "stream": False, **kwargs}

        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60,  # Increased from 30 to 60 seconds
            )
            response.raise_for_status()

            data = response.json()
            return OllamaResponse(
                response=data.get("response", ""),
                model=data.get("model", model),
                done=data.get("done", True),
                total_duration=data.get("total_duration", 0),
                load_duration=data.get("load_duration", 0),
                prompt_eval_count=data.get("prompt_eval_count", 0),
                eval_count=data.get("eval_count", 0),
            )

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return OllamaResponse(response=f"Error: {str(e)}", model=model, done=True)

    def list_models(self) -> List[str]:
        """List available models."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []


__all__ = ["OllamaClient", "OllamaResponse"]
