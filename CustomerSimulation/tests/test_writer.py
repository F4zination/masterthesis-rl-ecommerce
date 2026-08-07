import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


CUSTOMER_SIMULATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CUSTOMER_SIMULATION))

from simulation.writer import DatasetWriter  # noqa: E402


def _transition(
    trajectory_id: str,
    t: int,
    user_type: str,
    reward: float,
    events: list[str],
) -> dict:
    state = {"decision_point": "landing", "schema_version": 2}
    return {
        "trajectory_id": trajectory_id,
        "t": t,
        "decision_id": t,
        "timestamp": f"2026-01-01T00:00:{t:02d}",
        "session_id": trajectory_id,
        "decision_point": "landing",
        "state": state,
        "action": "no-op",
        "propensity": 1.0,
        "eligible_actions": ["no-op"],
        "reward": reward,
        "reward_without_cost": reward,
        "action_cost": 0.0,
        "next_state": state,
        "done": True,
        "metadata": {
            "user_type": user_type,
            "generated_events": events,
        },
    }


class DatasetWriterTests(unittest.TestCase):
    def test_persists_user_type_and_reports_per_archetype_session_metrics(self) -> None:
        sessions = [
            [_transition("explorer-1", 0, "Explorer", 1.0, ["purchase"])],
            [
                _transition("buyer-1", 0, "FastBuyer", 0.5, []),
                _transition("buyer-1", 1, "FastBuyer", -0.2, []),
            ],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path, csv_path, summary_path = DatasetWriter().write(
                sessions,
                tmp,
                {"Explorer": 1, "FastBuyer": 1, "WindowShopper": 0},
            )

            jsonl_rows = [
                json.loads(line)
                for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(jsonl_rows[0]["user_type"], "Explorer")
            self.assertEqual(jsonl_rows[1]["user_type"], "FastBuyer")
            self.assertNotIn("metadata", jsonl_rows[0])

            with csv_path.open(newline="", encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(csv_rows[0]["user_type"], "Explorer")
            self.assertEqual(csv_rows[1]["user_type"], "FastBuyer")

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["dataset_contract_version"], 1)
            self.assertEqual(len(summary["jsonl_sha256"]), 64)
            explorer = summary["per_archetype"]["Explorer"]
            buyer = summary["per_archetype"]["FastBuyer"]
            empty = summary["per_archetype"]["WindowShopper"]
            self.assertEqual(explorer["conversion_rate"], 1.0)
            self.assertEqual(explorer["mean_reward_per_session"], 1.0)
            self.assertEqual(buyer["n_transitions"], 2)
            self.assertAlmostEqual(buyer["mean_reward_per_session"], 0.3)
            self.assertEqual(buyer["avg_session_length"], 2.0)
            self.assertEqual(empty["n_sessions"], 0)

    def test_accepts_existing_top_level_user_type_without_metadata(self) -> None:
        transition = _transition("legacy-1", 0, "Explorer", 0.0, [])
        transition["user_type"] = "DetailedComparator"
        transition.pop("metadata")
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path, _, _ = DatasetWriter().write(
                [[transition]],
                tmp,
                {"DetailedComparator": 1},
            )
            row = json.loads(jsonl_path.read_text(encoding="utf-8"))
            self.assertEqual(row["user_type"], "DetailedComparator")

    def test_failed_jsonl_write_preserves_existing_dataset(self) -> None:
        transition = _transition("new-1", 0, "Explorer", 0.0, [])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            existing = output / "offline_transitions.jsonl"
            existing.write_text('{"existing": true}\n', encoding="utf-8")

            with (
                patch(
                    "simulation.writer.json.dumps",
                    side_effect=RuntimeError("simulated interruption"),
                ),
                self.assertRaisesRegex(RuntimeError, "simulated interruption"),
            ):
                DatasetWriter()._write_jsonl([transition], output)

            self.assertEqual(
                existing.read_text(encoding="utf-8"),
                '{"existing": true}\n',
            )
            self.assertEqual(
                list(output.glob(".offline_transitions.jsonl.*.tmp")),
                [],
            )


if __name__ == "__main__":
    unittest.main()
