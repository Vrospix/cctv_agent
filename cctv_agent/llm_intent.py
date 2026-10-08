"""Optional semantic normalisation through a small local Ollama model."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .models import QueryConstraints


DEFAULT_MODEL = "qwen2.5:1.5b"
DEFAULT_ENDPOINT = "http://127.0.0.1:11434/api/chat"
INTENTS = {"query", "control", "unknown"}

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": sorted(INTENTS)},
        "meaning": {"type": "string"},
    },
    "required": ["intent", "meaning"],
    "additionalProperties": False,
}


class IntentParserError(RuntimeError):
    """Raised when the optional local model cannot provide a safe intent."""


class IntentParser(Protocol):
    def interpret(
        self,
        text: str,
        *,
        today: date,
        context: QueryConstraints | None,
    ) -> "StructuredIntent": ...


@dataclass(frozen=True)
class StructuredIntent:
    """A small semantic envelope that the deterministic planner can validate."""

    intent: str
    meaning: str

    @classmethod
    def from_dict(cls, value: object) -> "StructuredIntent":
        if not isinstance(value, dict) or set(value) != {"intent", "meaning"}:
            raise IntentParserError("The model response did not match the intent schema.")
        intent = value["intent"]
        meaning = value["meaning"]
        if not isinstance(intent, str) or intent not in INTENTS:
            raise IntentParserError("The model returned an unsupported intent.")
        if not isinstance(meaning, str):
            raise IntentParserError("The model meaning must be text.")
        meaning = " ".join(meaning.split())
        if not meaning or len(meaning) > 180:
            raise IntentParserError("The model returned an invalid intent meaning.")
        return cls(intent=intent, meaning=meaning)

    def canonical_request(self) -> str:
        return "" if self.intent == "unknown" else self.meaning


class OllamaIntentParser:
    """Ask a local model for a labelled meaning, never executable SQL."""

    def __init__(
        self,
        *,
        model: str | None = None,
        endpoint: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.model = model or os.getenv("CCTV_OLLAMA_MODEL", DEFAULT_MODEL)
        self.endpoint = endpoint or os.getenv("CCTV_OLLAMA_URL", DEFAULT_ENDPOINT)
        self.timeout = timeout

    def interpret(
        self,
        text: str,
        *,
        today: date,
        context: QueryConstraints | None,
    ) -> StructuredIntent:
        payload = {
            "model": self.model,
            "stream": False,
            "format": INTENT_SCHEMA,
            "options": {"temperature": 0, "num_predict": 120},
            "messages": [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": text},
            ],
        }
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            content = result["message"]["content"]
            return StructuredIntent.from_dict(json.loads(content))
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            raise IntentParserError("The local Ollama service is unavailable.") from error
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise IntentParserError("The local model returned invalid structured JSON.") from error


def _system_prompt() -> str:
    return (
        "Return JSON that converts one CCTV message into a short command. intent is "
        "query, control, or unknown. Correct typos and slang. Preserve every camera, "
        "date, time, number, sort direction, and negation; add nothing. Use planner "
        "words such as show frames, count frames, previous results, exclude, yesterday, "
        "morning, and oldest. Example: 'how much footage did CTE get yday' means "
        "'count CTE frames yesterday'. 'Go one batch backward' means 'previous results'. "
        "Never answer or generate SQL."
    )
