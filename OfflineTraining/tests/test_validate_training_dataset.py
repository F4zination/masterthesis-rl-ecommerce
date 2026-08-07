from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from OfflineTraining.validate_training_dataset import (
    DatasetValidationError,
    validate_training_dataset,
)


def _state() -> dict:
    return {
        "decision_point": "landing",
        "schema_version": 2,
        "device_type": "unknown",
        "traffic_source": "direct",
        "page_depth_bucket": 1,
        "interventions_shown_bucket": 0,
        "steps_since_widget_bucket": 0,
        "session_step_bucket": 0,
        "primed_credit_bucket": 0,
    }


def _row(step: int, *, done: bool) -> dict:
    return {
        "trajectory_id": "session-1",
        "t": step,
        "decision_id": step,
        "timestamp": f"2026-01-01T00:00:{step:02d}",
        "session_id": "session-1",
        "decision_point": "landing",
        "state": _state(),
        "action": "no-op",
        "propensity": 1.0,
        "eligible_actions": [
            "no-op",
            "trending_carousel",
            "discount_banner",
            "frequently_bought_together",
            "trust_badge",
            "help_popup",
        ],
        "reward": 0.0,
        "reward_without_cost": 0.0,
        "action_cost": 0.0,
        "next_state": _state(),
        "done": done,
        "user_type": "Explorer",
    }


def _write_dataset(root: Path, rows: list[dict]) -> tuple[Path, Path]:
    dataset = root / "offline_transitions.jsonl"
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    dataset.write_text(payload, encoding="utf-8")
    summary = root / "dataset_summary.json"
    summary.write_text(
        json.dumps(
            {
                "dataset_contract_version": 1,
                "jsonl_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "n_transitions": len(rows),
                "n_sessions": 1,
                "archetype_distribution": {"Explorer": 1},
            }
        ),
        encoding="utf-8",
    )
    return dataset, summary


class TrainingDatasetValidationTests(unittest.TestCase):
    def test_accepts_complete_sequential_dataset_and_exact_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset, summary = _write_dataset(
                Path(tmp),
                [_row(0, done=False), _row(1, done=True)],
            )
            report = validate_training_dataset(
                dataset,
                summary_path=summary,
                expected_sessions=1,
                require_sequential=True,
            )

        self.assertTrue(report["valid"])
        self.assertEqual(report["n_sessions"], 1)
        self.assertEqual(report["n_transitions"], 2)

    def test_rejects_interrupted_dataset_even_when_file_is_nonempty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset, summary = _write_dataset(
                Path(tmp),
                [_row(0, done=False), _row(1, done=True)],
            )
            dataset.write_text(
                json.dumps(_row(0, done=False)) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                DatasetValidationError,
                "unfinished sessions",
            ):
                validate_training_dataset(
                    dataset,
                    summary_path=summary,
                    require_sequential=True,
                )

    def test_rejects_summary_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset, summary = _write_dataset(
                Path(tmp),
                [_row(0, done=True)],
            )
            payload = json.loads(summary.read_text(encoding="utf-8"))
            payload["jsonl_sha256"] = "0" * 64
            summary.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                DatasetValidationError,
                "jsonl_sha256 does not match",
            ):
                validate_training_dataset(dataset, summary_path=summary)

    def test_rejects_missing_sequential_feature(self) -> None:
        row = _row(0, done=True)
        row["state"].pop("primed_credit_bucket")
        with tempfile.TemporaryDirectory() as tmp:
            dataset, summary = _write_dataset(Path(tmp), [row])
            with self.assertRaisesRegex(
                DatasetValidationError,
                "missing sequential features",
            ):
                validate_training_dataset(
                    dataset,
                    summary_path=summary,
                    require_sequential=True,
                )


if __name__ == "__main__":
    unittest.main()
