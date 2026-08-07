from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from Experiments.calculate_clickworker_power import (
    calculate,
    mean_mde,
    required_per_arm,
)


class ClickworkerPowerTests(unittest.TestCase):
    def test_corrected_sample_sizes_and_reward_mdes(self):
        self.assertEqual(math.ceil(required_per_arm(0.08, 0.18)), 177)
        self.assertEqual(math.ceil(required_per_arm(0.25, 0.35)), 329)
        self.assertAlmostEqual(mean_mde(3.0, 150), 0.9706, places=3)
        self.assertAlmostEqual(mean_mde(3.0, 200), 0.8405, places=3)

    def test_equivalence_power_uses_all_ten_cells(self):
        cells = []
        for index in range(10):
            cells.append({
                "n_sessions": 2000,
                "conversion_rate": 0.08,
                "step_count_sd": 2.0,
                "funnel_depth_sd": 1.0,
            })
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "baseline.json"
            path.write_text(json.dumps({"cells": cells}), encoding="utf-8")
            result = calculate(path)
        conversion = result["primary_equivalence"]["30"]["conversion"]
        self.assertGreater(conversion["approximate_endpoint_power"], 0.85)
        self.assertLess(conversion["expected_ci90_half_width"], 0.05)


if __name__ == "__main__":
    unittest.main()
