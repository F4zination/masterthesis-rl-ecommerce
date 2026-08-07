#!/usr/bin/env python3
"""Generate a reproducible per-archetype face-validity table.

The audit deliberately uses the uniform behavior policy and switches all three
sequential mechanism strengths off.  It describes the hand-authored archetypes
without conflating their baseline magnitudes with either frozen study policy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "CustomerSimulation"))

from simulation.archetype import load_archetypes  # noqa: E402
from simulation.behavior_policy import BehaviorPolicy  # noqa: E402
from simulation.simulator import SessionSimulator, merge_dynamics  # noqa: E402


DEFAULT_OUT = REPO_ROOT / "Experiments" / "archetype_face_validity_20260721.json"
FUNNEL_DEPTH = {"landing": 1, "pdp": 2, "cart": 3, "checkout": 4}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    value_mean = sum(values) / len(values)
    return math.sqrt(
        sum((value - value_mean) ** 2 for value in values) / (len(values) - 1)
    )


def _converted(transitions: list[dict[str, Any]]) -> bool:
    return any(
        "purchase" in (transition.get("metadata", {}).get("generated_events") or [])
        for transition in transitions
    )


def _funnel_depth(transitions: list[dict[str, Any]]) -> float:
    if not transitions:
        return 0.0
    if _converted(transitions):
        return 5.0
    return float(
        max(FUNNEL_DEPTH.get(str(row.get("decision_point") or ""), 0) for row in transitions)
    )


def audit(
    config_path: Path,
    *,
    sessions_per_archetype: int,
    base_seed: int,
) -> dict[str, Any]:
    if sessions_per_archetype < 2:
        raise ValueError("sessions_per_archetype must be at least two")
    archetypes, mixture_prior, full_config = load_archetypes(str(config_path))
    dynamics = merge_dynamics(full_config.get("dynamics", {}) or {})
    dynamics.update(
        fatigue_rate=0.0,
        delayed_reward_strength=0.0,
        transition_coupling_strength=0.0,
    )
    t_max = int((full_config.get("simulation") or {}).get("t_max", 20))
    simulator = SessionSimulator(dynamics=dynamics, sequential=True)
    policy = BehaviorPolicy(epsilon=1.0)

    cells: list[dict[str, Any]] = []
    for index, (name, archetype) in enumerate(archetypes.items()):
        seed = int(base_seed) + (index + 1) * 100_003
        rng = random.Random(seed)
        converted: list[float] = []
        steps: list[float] = []
        rewards: list[float] = []
        depths: list[float] = []
        for _ in range(sessions_per_archetype):
            transitions = simulator.simulate_session(
                archetype, policy, t_max=t_max, rng=rng
            )
            converted.append(float(_converted(transitions)))
            steps.append(float(len(transitions)))
            rewards.append(sum(float(row["reward"]) for row in transitions))
            depths.append(_funnel_depth(transitions))
        cells.append(
            {
                "archetype": name,
                "seed": seed,
                "n_sessions": sessions_per_archetype,
                "conversion_rate": sum(converted) / len(converted),
                "conversion_count": int(sum(converted)),
                "mean_transitions": sum(steps) / len(steps),
                "transition_sd": _sample_sd(steps),
                "mean_reward": sum(rewards) / len(rewards),
                "reward_sd": _sample_sd(rewards),
                "mean_funnel_depth": sum(depths) / len(depths),
                "funnel_depth_sd": _sample_sd(depths),
            }
        )

    by_name = {cell["archetype"]: cell for cell in cells}
    mixture_total = sum(float(mixture_prior.get(name, 0.0)) for name in by_name)
    if mixture_total <= 0:
        raise ValueError("mixture_prior must have positive total weight")

    def weighted(metric: str) -> float:
        return sum(
            float(mixture_prior.get(name, 0.0)) * float(cell[metric])
            for name, cell in by_name.items()
        ) / mixture_total

    explorer_fast_total = float(mixture_prior.get("Explorer", 0.0)) + float(
        mixture_prior.get("FastBuyer", 0.0)
    )
    explorer_fast_conversion = (
        (
            float(mixture_prior.get("Explorer", 0.0))
            * float(by_name["Explorer"]["conversion_rate"])
            + float(mixture_prior.get("FastBuyer", 0.0))
            * float(by_name["FastBuyer"]["conversion_rate"])
        )
        / explorer_fast_total
        if explorer_fast_total
        else None
    )
    return {
        "schema_version": 1,
        "purpose": "face-validity magnitude audit; not empirical calibration",
        "design": {
            "policy": "uniform behavior policy",
            "sequential_features_emitted": True,
            "mechanism_strengths": {
                "fatigue_rate": 0.0,
                "delayed_reward_strength": 0.0,
                "transition_coupling_strength": 0.0,
            },
            "t_max": t_max,
            "sessions_per_archetype": sessions_per_archetype,
            "base_seed": base_seed,
        },
        "inputs": {
            "archetype_config": {
                "path": config_path.resolve().relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(config_path),
            },
            "script_sha256": _sha256(Path(__file__)),
        },
        "mixture_prior": mixture_prior,
        "configured_mixture": {
            "conversion_rate": weighted("conversion_rate"),
            "mean_transitions": weighted("mean_transitions"),
            "mean_reward": weighted("mean_reward"),
            "mean_funnel_depth": weighted("mean_funnel_depth"),
        },
        "configured_explorer_fastbuyer_submixture_conversion": explorer_fast_conversion,
        "cells": cells,
        "interpretation_boundary": (
            "Rank ordering and internal consistency are face-validity evidence only; "
            "the magnitudes and mixture are not calibrated to organic human traffic."
        ),
    }


def write_outputs(payload: dict[str, Any], json_path: Path) -> tuple[Path, Path]:
    csv_path = json_path.with_suffix(".csv")
    for path in (json_path, csv_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = payload["cells"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-per-archetype", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "CustomerSimulation" / "config" / "archetypes.yaml",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.sessions_per_archetype < 2:
        parser.error("--sessions-per-archetype must be at least two")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.out.name):
        parser.error("--out filename contains unsupported characters")
    payload = audit(
        args.config.resolve(),
        sessions_per_archetype=args.sessions_per_archetype,
        base_seed=args.seed,
    )
    json_path, csv_path = write_outputs(payload, args.out.resolve())
    print("ARCHETYPE              CONVERSION  TRANSITIONS  REWARD  FUNNEL")
    for cell in payload["cells"]:
        print(
            f"{cell['archetype']:<22}{cell['conversion_rate']:>10.2%}"
            f"{cell['mean_transitions']:>13.3f}{cell['mean_reward']:>8.3f}"
            f"{cell['mean_funnel_depth']:>8.3f}"
        )
    print(f"Configured mixture conversion: {payload['configured_mixture']['conversion_rate']:.2%}")
    print(f"Explorer/FastBuyer sub-mixture: {payload['configured_explorer_fastbuyer_submixture_conversion']:.2%}")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
