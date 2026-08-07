#!/usr/bin/env python3
"""Null-calibration floors for the Section 7.1 cell-level calibration endpoint.

``ClickworkerPreregistration.md`` §7.1 makes the mean absolute cell error (MACE)
a secondary calibration endpoint, and requires its *null-calibration floor* to be
reported beside every observed value. The floor exists because MACE is biased
upward: since E|X| >= |EX|, sampling noise alone produces a positive MACE even
when every cell is perfectly calibrated. A tolerance that does not sit clearly
above the floor cannot detect miscalibration, and the endpoint must then be
reported as inconclusive rather than as a pass.

This script computes those floors from a locked simulator reference, so the
planning table in §7.1 is reproducible from an artifact rather than asserted.
The preregistration also requires recomputing them against the locked study
release before recruitment, and again at the achieved sample size before
unblinding.

Method, following §7.1:

* For metric ``m`` and cell ``(p,a)`` the discrepancy is the human cell mean
  minus the context-standardized simulator cell mean. Under the null both are
  estimates of the same quantity, so the discrepancy is centred at zero with
  variance contributed by two independent sources.
* Human sampling error is ``sd / sqrt(n_per_cell)``. Human cell variance is
  assumed equal to the simulator cell variance, the same planning assumption
  ``calculate_clickworker_power.py`` records as ``equal_to_simulator_cell_sd``.
* Simulator Monte Carlo error is ``sd / sqrt(n_sessions)`` at the reference's
  own sessions per cell.
* Conversion is Bernoulli, so its cell ``sd`` is ``sqrt(p(1-p))`` rather than a
  stored moment.
* Drawing one discrepancy per cell from ``Normal(0, se_cell)`` and averaging the
  absolute values gives one parametric-bootstrap replicate of MACE under the
  null. The floor is the mean of that distribution and ``h0_upper`` its
  one-sided 95% limit -- the value an observed upper limit must exceed for a
  failure to mean anything.

Usage::

    python Experiments/calculate_mace_floors.py \
        --baseline Experiments/study_policy_baselines/<id>/policy_archetype_baseline.json \
        --out Experiments/study_policy_baselines/<id>/mace_floors.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

# Endpoint name -> (baseline mean field, baseline sd field). Conversion carries
# no stored sd; its Bernoulli sd is derived from the rate.
ENDPOINTS: dict[str, tuple[str, str | None]] = {
    "conversion": ("conversion_rate", None),
    "transition_count": ("step_count_mean", "step_count_sd"),
    "funnel_depth": ("funnel_depth_mean", "funnel_depth_sd"),
}

# §7.1 tolerances. Substantive and fixed; not revisable after seeing outcomes.
TOLERANCES = {
    "conversion": 0.10,
    "transition_count": 1.00,
    "funnel_depth": 1.00,
}

DEFAULT_N_PER_CELL = (30, 40)


def cell_sd(cell: dict, endpoint: str) -> float:
    """Return the simulator cell standard deviation for one endpoint.

    Args:
        cell: One entry of the baseline's ``cells`` list.
        endpoint: Key of :data:`ENDPOINTS`.

    Returns:
        The standard deviation of the metric within that cell.
    """
    mean_field, sd_field = ENDPOINTS[endpoint]
    if sd_field is None:
        p = float(cell[mean_field])
        return math.sqrt(max(p * (1.0 - p), 0.0))
    return float(cell[sd_field])


def cell_standard_errors(cells: list[dict], endpoint: str, n_per_cell: int) -> list[float]:
    """Combine human sampling error and simulator Monte Carlo error per cell.

    Args:
        cells: The baseline's ten policy x archetype cells.
        endpoint: Key of :data:`ENDPOINTS`.
        n_per_cell: Human participants per randomized cell.

    Returns:
        One standard error per cell, in the order given.
    """
    errors = []
    for cell in cells:
        sd = cell_sd(cell, endpoint)
        se_human = sd / math.sqrt(n_per_cell)
        se_simulator = sd / math.sqrt(float(cell["n_sessions"]))
        errors.append(math.hypot(se_human, se_simulator))
    return errors


def null_mace_distribution(
    standard_errors: list[float], replicates: int, rng: random.Random
) -> list[float]:
    """Draw MACE replicates under the null of zero true discrepancy everywhere.

    Args:
        standard_errors: Per-cell standard errors of the discrepancy.
        replicates: Number of parametric-bootstrap replicates.
        rng: Seeded random source.

    Returns:
        ``replicates`` sampled values of the mean absolute cell error.
    """
    n_cells = len(standard_errors)
    draws = []
    for _ in range(replicates):
        total = 0.0
        for se in standard_errors:
            total += abs(rng.gauss(0.0, se))
        draws.append(total / n_cells)
    return draws


def _percentile(sorted_values: list[float], q: float) -> float:
    """Return the ``q`` quantile of an ascending list by linear interpolation."""
    if not sorted_values:
        return float("nan")
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def floors_for(
    cells: list[dict], endpoint: str, n_per_cell: int, replicates: int, seed: int
) -> dict:
    """Compute the floor and one-sided 95% null limit for one endpoint.

    Args:
        cells: The baseline's ten cells.
        endpoint: Key of :data:`ENDPOINTS`.
        n_per_cell: Human participants per randomized cell.
        replicates: Parametric-bootstrap replicates.
        seed: Base seed; mixed with endpoint and sample size so each
            configuration draws independently but reproducibly.

    Returns:
        A dict of the floor, its analytic counterpart, the null upper limit,
        the tolerance and the headroom.
    """
    errors = cell_standard_errors(cells, endpoint, n_per_cell)
    mixed = hashlib.sha256(f"{seed}:{endpoint}:{n_per_cell}".encode()).hexdigest()
    rng = random.Random(int(mixed[:16], 16))

    draws = sorted(null_mace_distribution(errors, replicates, rng))
    floor = statistics.fmean(draws)
    # E|X| for X ~ Normal(0, s) is s * sqrt(2/pi); averaging over cells gives the
    # floor in closed form, which is a check on the sampled value.
    analytic = statistics.fmean(errors) * math.sqrt(2.0 / math.pi)
    h0_upper = _percentile(draws, 0.95)
    tolerance = TOLERANCES[endpoint]

    return {
        "n_per_cell": n_per_cell,
        "floor": floor,
        "floor_analytic": analytic,
        "h0_upper": h0_upper,
        "tolerance": tolerance,
        "headroom_vs_h0_upper": tolerance / h0_upper if h0_upper else float("inf"),
        "cell_standard_errors": errors,
        "assumed_human_sd": "equal_to_simulator_cell_sd",
        "assumed_true_gap": 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="policy_archetype_baseline.json from the locked simulator reference.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path.")
    parser.add_argument(
        "--n-per-cell",
        type=int,
        nargs="+",
        default=list(DEFAULT_N_PER_CELL),
        help="Participants per randomized cell (default: 30 40).",
    )
    parser.add_argument("--replicates", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--personas",
        type=str,
        default="",
        help=(
            "Comma-separated archetype names to restrict the MACE cells to "
            "(e.g. the recruited subset of a design that assigns fewer "
            "personas than the reference evaluates). Empty means all cells."
        ),
    )
    args = parser.parse_args()

    if args.replicates < 1:
        parser.error("--replicates must be positive")
    if any(n < 1 for n in args.n_per_cell):
        parser.error("--n-per-cell values must be positive")

    payload = json.loads(args.baseline.read_text(encoding="utf-8"))
    cells = payload["cells"]
    personas = [item.strip() for item in args.personas.split(",") if item.strip()]
    if personas:
        available = {str(cell.get("archetype")) for cell in cells}
        unknown = sorted(set(personas) - available)
        if unknown:
            parser.error(f"personas not in baseline: {unknown}; available: {sorted(available)}")
        cells = [cell for cell in cells if str(cell.get("archetype")) in personas]
    if len(cells) % 2 != 0 or not cells:
        parser.error(f"expected an even, non-empty policy x archetype cell count, found {len(cells)}")

    digest = hashlib.sha256(args.baseline.read_bytes()).hexdigest()

    results: dict[str, dict[str, dict]] = {}
    for endpoint in ENDPOINTS:
        results[endpoint] = {
            str(n): floors_for(cells, endpoint, n, args.replicates, args.seed)
            for n in args.n_per_cell
        }

    out = {
        "purpose": (
            "Null-calibration floors for the ClickworkerPreregistration.md §7.1 "
            "cell-level (MACE) calibration endpoint."
        ),
        "method": (
            "Parametric bootstrap under zero true discrepancy in every cell. "
            "Per-cell SE combines human sampling error (sd/sqrt(n_per_cell), human "
            "variance assumed equal to the simulator cell variance) with simulator "
            "Monte Carlo error (sd/sqrt(n_sessions)). Conversion uses the Bernoulli "
            "sd sqrt(p(1-p))."
        ),
        "baseline": {
            "path": str(args.baseline),
            "sha256": digest,
            "study_id": payload.get("study_id"),
            "git_head": payload.get("git_head"),
            "sessions_per_cell": cells[0]["n_sessions"],
            "personas": personas or sorted({str(cell.get("archetype")) for cell in cells}),
            "n_cells": len(cells),
        },
        "replicates": args.replicates,
        "seed": args.seed,
        "endpoints": results,
        "interpretation": (
            "The floor is the MACE expected from sampling noise alone under perfect "
            "calibration; h0_upper is the one-sided 95% limit under that null. A "
            "tolerance must sit clearly above h0_upper for the endpoint to be able "
            "to detect miscalibration; otherwise report it as inconclusive rather "
            "than as a pass. Recompute at the achieved sample size before unblinding."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Baseline: {payload.get('study_id')}  ({digest[:16]})")
    print(f"Replicates: {args.replicates}  seed: {args.seed}\n")
    header = f"{'Endpoint':<18}{'n/cell':>7}{'Floor':>9}{'H0 upper':>10}{'Tolerance':>11}{'Headroom':>10}"
    print(header)
    print("-" * len(header))
    for endpoint, by_n in results.items():
        for n, values in by_n.items():
            print(
                f"{endpoint:<18}{n:>7}{values['floor']:>9.3f}"
                f"{values['h0_upper']:>10.3f}{values['tolerance']:>11.2f}"
                f"{values['headroom_vs_h0_upper']:>9.1f}x"
            )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
