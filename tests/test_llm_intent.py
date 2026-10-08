from __future__ import annotations

import json
import unittest
from datetime import date
from unittest.mock import patch

from cctv_agent.agent import CCTVQueryAgent
from cctv_agent.database import FrameDatabase
from cctv_agent.llm_intent import IntentParserError, OllamaIntentParser, StructuredIntent


class FakeIntentParser:
    def __init__(self, result: StructuredIntent) -> None:
        self.result = result
        self.calls = 0

    def interpret(self, text, *, today, context):
        self.calls += 1
        return self.result


class FakeHTTPResponse:
    def __init__(self, value: dict[str, object]) -> None:
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.value).encode("utf-8")


class StructuredIntentTests(unittest.TestCase):
    def test_structured_intent_exposes_normalized_meaning(self) -> None:
        intent = StructuredIntent(intent="query", meaning="show CTE frames yesterday")

        self.assertEqual("show CTE frames yesterday", intent.canonical_request())

    def test_unknown_intent_has_no_canonical_request(self) -> None:
        intent = StructuredIntent(intent="unknown", meaning="show me the relevant one")

        self.assertEqual("", intent.canonical_request())

    def test_extra_structured_field_is_rejected(self) -> None:
        with self.assertRaises(IntentParserError):
            StructuredIntent.from_dict(
                {"intent": "query", "meaning": "show frames", "sql": "SELECT *"}
            )

    def test_ollama_response_is_read_as_structured_json(self) -> None:
        content = {"intent": "query", "meaning": "count PIE frames today"}
        response = FakeHTTPResponse({"message": {"content": json.dumps(content)}})

        with patch("cctv_agent.llm_intent.urlopen", return_value=response) as call:
            parsed = OllamaIntentParser(timeout=1).interpret(
                "how much PIE footage is there now",
                today=date(2026, 9, 15),
                context=None,
            )

        request = call.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual("query", parsed.intent)
        self.assertEqual("count PIE frames today", parsed.meaning)
        self.assertEqual(["intent", "meaning"], payload["format"]["required"])
        self.assertFalse(payload["stream"])

    def test_agent_uses_local_ai_only_as_unknown_intent_fallback(self) -> None:
        database = FrameDatabase()
        database.connection.execute(
            "INSERT INTO frames(captured_at, camera_name) VALUES (?, ?)",
            ("2026-09-14T08:00:00", "Central Expressway"),
        )
        parser = FakeIntentParser(
            StructuredIntent(intent="query", meaning="show frames yesterday")
        )
        agent = CCTVQueryAgent(
            database,
            today=date(2026, 9, 15),
            intent_parser=parser,
        )

        result = agent.ask("plz zorp the relevant thing the other day")

        self.assertEqual(1, result.total)
        self.assertEqual("local_ai", agent.last_intent_source)
        self.assertEqual(1, parser.calls)

    def test_model_cannot_invent_a_camera(self) -> None:
        database = FrameDatabase()
        parser = FakeIntentParser(
            StructuredIntent(intent="query", meaning="show PIE frames today")
        )
        agent = CCTVQueryAgent(
            database,
            today=date(2026, 9, 15),
            intent_parser=parser,
        )

        result = agent.ask("show me the relevant one")

        self.assertEqual("deterministic", agent.last_intent_source)
        self.assertIn("only help search CCTV frames", result.message)
        self.assertIn("added a camera", agent.last_llm_error or "")

    def test_model_can_normalize_an_unusual_page_control(self) -> None:
        database = FrameDatabase()
        parser = FakeIntentParser(
            StructuredIntent(intent="control", meaning="previous results")
        )
        agent = CCTVQueryAgent(
            database,
            today=date(2026, 9, 15),
            intent_parser=parser,
        )
        agent.ask("show ECP frames today")

        result = agent.ask("return to the prior batch")

        self.assertEqual("local_ai", agent.last_intent_source)
        self.assertIn("first page", result.message)

    def test_known_deterministic_query_does_not_call_local_ai(self) -> None:
        database = FrameDatabase()
        parser = FakeIntentParser(
            StructuredIntent(intent="unknown", meaning="show ECP frames today")
        )
        agent = CCTVQueryAgent(
            database,
            today=date(2026, 9, 15),
            intent_parser=parser,
        )

        agent.ask("Show ECP frames today")

        self.assertEqual(0, parser.calls)
        self.assertEqual("deterministic", agent.last_intent_source)


if __name__ == "__main__":
    unittest.main()
