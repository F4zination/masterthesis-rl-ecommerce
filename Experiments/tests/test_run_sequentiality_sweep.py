from __future__ import annotations

import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from Experiments import run_sequentiality_sweep as sweep


class SequentialitySweepDurabilityTests(unittest.TestCase):
    def test_five_seed_ci_uses_student_t_not_normal_critical_value(self) -> None:
        values = [0.0, 1.0, 2.0, 3.0, 4.0]
        mean = sum(values) / len(values)
        sd = math.sqrt(
            sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        )

        half_width = sweep._ci95(values)

        self.assertAlmostEqual(half_width, 2.776 * sd / math.sqrt(5))
        self.assertGreater(half_width, 1.96 * sd / math.sqrt(5))

    def test_axis_value_overrides_validate_and_preserve_t_max_as_int(self) -> None:
        grid = {axis: list(values) for axis, values in sweep.DEFAULT_AXES.items()}

        sweep.apply_axis_value_overrides(
            grid,
            ["fatigue_rate=0,0.6", "t_max=20,60"],
        )

        self.assertEqual(grid["fatigue_rate"], [0.0, 0.6])
        self.assertEqual(grid["t_max"], [20, 60])
        self.assertTrue(all(isinstance(value, int) for value in grid["t_max"]))
        with self.assertRaises(SystemExit):
            sweep.apply_axis_value_overrides(grid, ["t_max=20.5"])

    def test_cell_cache_round_trip_and_fingerprint_guard(self) -> None:
        identity = sweep._cell_identity("fatigue_rate", 0.4, 7)
        stats = {
            "bandit_v2": {"mean_reward": 1.0},
            "ppo_v3": {"mean_reward": 1.2},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cell.json"
            sweep._save_cached_cell(path, identity, "fingerprint-a", stats)

            self.assertEqual(
                sweep._load_cached_cell(path, identity, "fingerprint-a"),
                stats,
            )
            with self.assertRaises(SystemExit):
                sweep._load_cached_cell(path, identity, "fingerprint-b")

    def test_run_manifest_rejects_mixed_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            spec = {"grid": [0.0, 1.0], "seed": 5}
            fingerprint, manifest = sweep._initialise_resume_manifest(out_dir, spec)

            self.assertTrue(manifest.exists())
            self.assertEqual(
                sweep._initialise_resume_manifest(out_dir, spec)[0], fingerprint,
            )
            with self.assertRaises(SystemExit):
                sweep._initialise_resume_manifest(
                    out_dir, {"grid": [0.0, 2.0], "seed": 5},
                )

    def test_independent_smoke_processes_produce_identical_seed_results(self) -> None:
        script = Path(sweep.__file__).resolve()
        repo_root = script.parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            outputs = [base / "run_a", base / "run_b"]
            for out_dir in outputs:
                subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        "--smoke",
                        "--out-dir",
                        str(out_dir),
                    ],
                    cwd=repo_root,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

            for artifact in ("long_results.csv", "gaps.csv", "deltas.csv"):
                self.assertEqual(
                    (outputs[0] / artifact).read_bytes(),
                    (outputs[1] / artifact).read_bytes(),
                    artifact,
                )


if __name__ == "__main__":
    unittest.main()
