"""The provider boundary: every LLM sits behind `LLMClient`, so adding a model touches no
other code. `OllamaClient` talks to a local server over HTTP using only the standard library.
"""

import json
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class LLMResponse:
    text: str
    tokens: int


class LLMClient(Protocol):
    model: str
    # Identifies the exact weights, so a re-pulled model under the same name changes the cache key.
    model_version: str

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
        """One chat turn constrained to `schema`. Raises on transport errors, never on bad JSON."""
        ...


class OllamaClient:
    def __init__(
        self,
        model: str,
        model_version: str,
        *,
        host: str = "http://localhost:11434",
        seed: int = 0,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.model_version = model_version
        self._host = host.rstrip("/")
        self._seed = seed
        self._timeout = timeout

    @classmethod
    def connect(cls, model: str, *, host: str = "http://localhost:11434") -> "OllamaClient":
        """Pin `model_version` to the digest of the locally pulled weights."""
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=10) as reply:
            tags = json.load(reply)
        for entry in tags["models"]:
            if model in (entry["name"], entry["model"]):
                return cls(model, entry["digest"], host=host)
        raise LookupError(f"{model} is not pulled; run `ollama pull {model}`")

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": schema,
            "stream": False,
            "think": False,  # no reasoning tokens: a classification needs none and they cost time
            "options": {"temperature": 0, "seed": self._seed},
        }
        request = urllib.request.Request(
            f"{self._host}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as reply:
            data = json.load(reply)
        tokens = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
        return LLMResponse(text=data["message"]["content"], tokens=tokens)
