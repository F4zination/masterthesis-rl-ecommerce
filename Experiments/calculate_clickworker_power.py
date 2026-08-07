#!/usr/bin/env python3
"""Reproduce the Clickworker study's planning approximations.

The calculations are deliberately simple, transparent normal approximations.
They are planning aids, not post-hoc tests and not substitutes for the locked
bootstrap/randomization analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Sequence


ALPHA = 0.05
POWER = 0.80
Z_TWO_SIDED = NormalDist().inv_cdf(1.0 - ALPHA / 2.0)
Z_POWER = NormalDist().inv_cdf(POWER)
Z_TOST = NormalDist().inv_cdf(1.0 - ALPHA)


def required_per_arm(p1: float, p2: float) -> float:
    """Two-sided equal-arm normal-approximation sample size."""
    pooled = (p1 + p2) / 2.0
    numerator = (
        Z_TWO_SIDED * math.sqrt(2.0 * pooled * (1.0 - pooled))
        + Z_POWER * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    ) ** 2
    return numerator / ((p2 - p1) ** 2)


def proportion_mde(p1: float, n_per_arm: int) -> float:
    """Smallest higher p2 whose planning sample size is at most n_per_arm."""
    lower, upper = p1, 1.0 - 1e-12
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if required_per_arm(p1, midpoint) > n_per_arm:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def mean_mde(sd: float, n_per_arm: int) -> float:
    return (Z_TWO_SIDED + Z_POWER) * sd * math.sqrt(2.0 / n_per_arm)


def _cell_sd(cell: dict[str, Any], outcome: str) -> float:
    if outcome == "conversion":
        n = int(cell["n_sessions"])
        p = float(cell["conversion_rate"])
        return math.sqrt(p * (1.0 - p) * n / (n - 1))
    if outcome == "session_length_steps":
        return float(cell["step_count_sd"])
    if outcome == "funnel_depth":
        return float(cell["funnel_depth_sd"])
    raise ValueError(outcome)


def equivalence_planning(
    baseline: dict[str, Any],
    *,
    n_per_cell: int,
    outcome: str,
    margin: float,
) -> dict[str, float]:
    """Endpointwise TOST power at zero bias and equal human/simulator SDs."""
    cells = baseline["cells"]
    human_variance = sum(_cell_sd(cell, outcome) ** 2 / n_per_cell for cell in cells)
    simulation_variance = sum(
        _cell_sd(cell, outcome) ** 2 / int(cell["n_sessions"])
        for cell in cells
    )
    standard_error = math.sqrt(human_variance + simulation_variance) / len(cells)
    ci_half_width = Z_TOST * standard_error
    threshold_z = (margin - ci_half_width) / standard_error
    pass_probability = max(
        0.0,
        2.0 * NormalDist().cdf(threshold_z) - 1.0,
    )
    return {
        "n_per_cell": n_per_cell,
        "assumed_true_gap": 0.0,
        "assumed_human_sd": "equal_to_simulator_cell_sd",
        "standard_error": standard_error,
        "expected_ci90_half_width": ci_half_width,
        "margin": margin,
        "approximate_endpoint_power": pass_probability,
    }


def calculate(baseline_path: Path) -> dict[str, Any]:
    baseline_bytes = baseline_path.read_bytes()
    baseline = json.loads(baseline_bytes.decode("utf-8"))
    margins = {
        "conversion": 0.05,
        "session_length_steps": 0.50,
        "funnel_depth": 0.50,
    }
    return {
        "method": "normal approximations for planning only",
        "alpha_two_sided": ALPHA,
        "target_power": POWER,
        "baseline": {
            "path": str(baseline_path),
            "sha256": hashlib.sha256(baseline_bytes).hexdigest(),
            "reference_scope": "ten aggregate policy-by-persona cells",
            "context_strata_present": bool(baseline.get("context_strata")),
        },
        "proportion_examples": [
            {
                "from": p1,
                "to": p2,
                "required_per_policy_unrounded": required_per_arm(p1, p2),
                "required_per_policy_ceiling": math.ceil(required_per_arm(p1, p2)),
            }
            for p1, p2 in ((0.08, 0.18), (0.08, 0.13), (0.08, 0.10), (0.25, 0.35))
        ],
        "ab_mde": [
            {
                "n_per_policy": n,
                "conversion_from": 0.08,
                "conversion_to": proportion_mde(0.08, n),
                "reward_sd": 3.0,
                "reward_mde": mean_mde(3.0, n),
            }
            for n in (150, 200)
        ],
        "primary_equivalence": {
            str(n): {
                outcome: equivalence_planning(
                    baseline,
                    n_per_cell=n,
                    outcome=outcome,
                    margin=margin,
                )
                for outcome, margin in margins.items()
            }
            for n in (30, 40)
        },
        "interpretation": (
            "Equivalence power assumes zero human-simulator bias and human cell variances "
            "equal to the simulator reference. It is endpointwise; joint power depends on "
            "unknown cross-endpoint dependence. It uses the aggregate-cell SDs as a "
            "pre-context approximation for the locked context-standardized analysis. "
            "Margins require substantive approval."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(
            "Experiments/study_policy_baselines/"
            "clickworker_pre_recruitment_20260721/policy_archetype_baseline.json"
        ),
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if not args.baseline.is_file():
        parser.error(f"baseline not found: {args.baseline}")
    payload = calculate(args.baseline)
    rendered = json.dumps(payload, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
