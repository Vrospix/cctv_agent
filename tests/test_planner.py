from __future__ import annotations

import json
import unittest
from datetime import date
from pathlib import Path

from cctv_agent.planner import build_query_plan


class PlannerEvaluationTests(unittest.TestCase):
    def test_language_evaluation_cases(self) -> None:
        path = Path(__file__).with_name("evaluation_cases.json")
        cases = json.loads(path.read_text(encoding="utf-8"))
        for case in cases:
            with self.subTest(text=case["input"]):
                plan = build_query_plan(
                    case["input"], today=date(2026, 9, 16), context=None
                )
                for field in (
                    "intent",
                    "camera_action",
                    "date_action",
                    "time_action",
                    "sort_order",
                    "limit",
                    "sample_action",
                    "sample_interval_minutes",
                    "weekday",
                ):
                    if field in case:
                        self.assertEqual(case[field], getattr(plan, field))
                if "camera" in case:
                    self.assertIn(case["camera"], plan.cameras)
                if "camera_count" in case:
                    self.assertEqual(case["camera_count"], len(plan.cameras))
                if "time_start" in case:
                    self.assertEqual(case["time_start"], plan.time_start.isoformat())
                if "time_end" in case:
                    self.assertEqual(case["time_end"], plan.time_end.isoformat())


if __name__ == "__main__":
    unittest.main()
