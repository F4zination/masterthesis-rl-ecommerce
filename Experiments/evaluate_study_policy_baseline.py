#!/usr/bin/env python3
"""Build the policy- and context-matched Clickworker simulator reference.

This evaluator uses the exact local artifacts selected by the study deployment:

* V2: deterministic greedy serving over ``DemoSiteV2/data/demosite.db``;
* V3: deterministic masked argmax from
  ``OfflineTraining/outputs/ppo_policy.pt``.

Every archetype receives the same number of sessions under each policy.  The
two aggregate policy cells for an archetype receive the same seed, which
provides a paired diagnostic without changing either policy.  Release runs
also evaluate every policy x archetype x device x traffic stratum using
independent deterministic seeds.  Human-to-simulator fidelity can therefore
standardize the simulator to the observed exogenous context mix rather than
the simulator's otherwise-uniform context generator.  Output is a
study-specific JSON manifest plus a flat aggregate-cell CSV table.  Existing
output directories are never overwritten; choose a new ``--study-id`` to
rerun.

Example::

    .venv/Scripts/python Experiments/evaluate_study_policy_baseline.py \
        --study-id clickworker_pre_recruitment_20260721 \
        --sessions-per-cell 2000 --context-sessions-per-stratum 500 \
        --seed 20260721
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import re
import sqlite3
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "OfflineTraining"))
sys.path.insert(0, str(REPO_ROOT / "CustomerSimulation"))
sys.path.insert(0, str(REPO_ROOT / "SharedSchema"))

from eval_all_policies_sim import (  # noqa: E402
    BanditPolicy as _BaseBanditPolicy,
    PPOPolicy as _BasePPOPolicy,
)
from simulation.archetype import Archetype, load_archetypes  # noqa: E402
from simulation.simulator import SessionSimulator, merge_dynamics  # noqa: E402
from shared_schema.constants import (  # noqa: E402
    ALL_ACTIONS,
    DEVICE_TYPES,
    FEATURE_BUCKET_THRESHOLDS,
    PRIOR_COUNT,
    PRIOR_MEAN,
    TRAFFIC_SOURCES,
)
from shared_schema.features import (  # noqa: E402
    POINT_FEATURES,
    _bucket,
    _history_from_actions,
)

try:  # Keep the error at the CLI boundary clear when Torch is unavailable.
    import torch
except ImportError:  # pragma: no cover - exercised only in a minimal runtime
    torch = None


JSON_NAME = "policy_archetype_baseline.json"
CSV_NAME = "policy_archetype_baseline.csv"
DEFAULT_STUDY_ID = "clickworker_pre_recruitment"
DEVICE_STRATA = DEVICE_TYPES
TRAFFIC_STRATA = TRAFFIC_SOURCES


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _input_descriptor(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": _display_path(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={REPO_ROOT.as_posix()}",
                "rev-parse",
                "HEAD",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def derive_archetype_seeds(archetype_names: list[str], base_seed: int) -> dict[str, int]:
    """Create stable, distinct seeds shared by the two policy conditions."""
    return {
        name: int(base_seed) + (index + 1) * 100_003
        for index, name in enumerate(archetype_names)
    }


def derive_context_seed(
    policy_index: int,
    archetype_index: int,
    device_index: int,
    traffic_index: int,
    base_seed: int,
) -> int:
    """Stable independent seed for one policy and exogenous-context stratum."""
    return (
        int(base_seed)
        + (policy_index + 1) * 10_000_019
        + (archetype_index + 1) * 100_003
        + (device_index + 1) * 1_009
        + (traffic_index + 1) * 97
    )


def _production_history_buckets(actions: list[str]) -> dict[str, int]:
    """Recreate DemoSiteV3's serving-time history normalization exactly.

    Production reconstructs ``primed_credit`` from every prior action even
    when the simulator's delayed-reward mechanism is disabled.  Recomputing
    all four fields here both matches the deployed PPO input and lets the OOV
    report expose checkpoint/serving drift that the simulator state alone
    would otherwise hide.
    """
    history = _history_from_actions(actions)
    return {
        "interventions_shown_bucket": _bucket(
            float(history["interventions_shown"]),
            FEATURE_BUCKET_THRESHOLDS["interventions_shown_bucket"],
        ),
        "steps_since_widget_bucket": _bucket(
            float(history["steps_since_intervention"]),
            FEATURE_BUCKET_THRESHOLDS["steps_since_widget_bucket"],
        ),
        "session_step_bucket": _bucket(
            float(history["session_step"]),
            FEATURE_BUCKET_THRESHOLDS["session_step_bucket"],
        ),
        "primed_credit_bucket": _bucket(
            float(history["primed_credit"]),
            FEATURE_BUCKET_THRESHOLDS["primed_credit_bucket"],
        ),
    }


class FrozenBanditPolicy(_BaseBanditPolicy):
    """V2's frozen greedy policy with read-only coverage diagnostics."""

    def __init__(self, db_path: Path) -> None:
        super().__init__(db_path)
        self.reset_diagnostics()

    def begin_session(self) -> None:
        """Policy is memoryless; hook retained for the shared evaluator."""

    def reset_diagnostics(self) -> None:
        self._decisions = 0
        self._missing_feature_decisions = 0
        self._unseen_context_decisions = 0
        self._partially_unseen_context_decisions = 0
        self._selected_unseen_arm_decisions = 0
        self._missing_features: Counter[str] = Counter()

    def select_action(
        self,
        eligible_actions: list[str],
        archetype_name: str | None = None,
        rng: random.Random | None = None,
    ) -> tuple[str, float]:
        self._decisions += 1
        state = self.current_state
        decision_point = str(state.get("decision_point", ""))
        features = POINT_FEATURES.get(decision_point)
        if features is None or any(feature not in state for feature in features):
            self._missing_feature_decisions += 1
            if features is None:
                self._missing_features[f"unknown_decision_point:{decision_point}"] += 1
            else:
                for feature in features:
                    if feature not in state:
                        self._missing_features[feature] += 1
            fallback = "no-op" if "no-op" in eligible_actions else eligible_actions[0]
            # Frozen V2 is deterministic, including its cold-context fallback.
            return fallback, 1.0

        context_key = "|".join(
            [decision_point] + [str(state[feature]) for feature in features]
        )
        known = [
            (decision_point, context_key, action) in self.arms
            for action in eligible_actions
        ]
        if not any(known):
            self._unseen_context_decisions += 1
        elif not all(known):
            self._partially_unseen_context_decisions += 1

        best_action = eligible_actions[0]
        best_score: float | None = None
        for action in eligible_actions:
            score = self._score(decision_point, context_key, action)
            if best_score is None or score > best_score:
                best_action = action
                best_score = score

        if (decision_point, context_key, best_action) not in self.arms:
            self._selected_unseen_arm_decisions += 1
        return best_action, 1.0

    def diagnostics(self) -> dict[str, Any]:
        denominator = max(self._decisions, 1)
        return {
            "decisions": self._decisions,
            "missing_feature_decisions": self._missing_feature_decisions,
            "unseen_context_decisions": self._unseen_context_decisions,
            "unseen_context_rate": self._unseen_context_decisions / denominator,
            "partially_unseen_context_decisions": self._partially_unseen_context_decisions,
            "selected_unseen_arm_decisions": self._selected_unseen_arm_decisions,
            "missing_features": dict(sorted(self._missing_features.items())),
            "fallback_decisions": self._missing_feature_decisions,
        }


class FrozenPPOPolicy(_BasePPOPolicy):
    """V3's frozen PPO argmax policy with serving-input OOV diagnostics."""

    def __init__(self, checkpoint_path: Path) -> None:
        super().__init__(checkpoint_path)
        self.checkpoint_path = checkpoint_path.resolve()
        self._prior_actions: list[str] = []
        self.reset_diagnostics()

    def begin_session(self) -> None:
        self._prior_actions = []

    def reset_diagnostics(self) -> None:
        self._decisions = 0
        self._fallback_decisions = 0
        self._decisions_with_oov = 0
        self._token_occurrences = 0
        self._oov_occurrences = 0
        self._oov_tokens: Counter[str] = Counter()
        self._policy_scores: list[float] = []

    def set_state(self, state_dict: dict[str, Any]) -> None:
        state = dict(state_dict)
        state.update(_production_history_buckets(self._prior_actions))
        self.current_state = state

    def select_action(
        self,
        eligible_actions: list[str],
        archetype_name: str | None = None,
        rng: random.Random | None = None,
    ) -> tuple[str, float]:
        if torch is None:  # pragma: no cover - constructor already fails first
            raise RuntimeError("PyTorch is required to evaluate the PPO checkpoint")

        self._decisions += 1
        state = self.current_state
        scalar_tokens = {
            f"{key}={value}"
            for key, value in state.items()
            if value is not None and not isinstance(value, (dict, list))
        }
        oov_tokens = sorted(token for token in scalar_tokens if token not in self.state_vocab)
        self._token_occurrences += len(scalar_tokens)
        self._oov_occurrences += len(oov_tokens)
        if oov_tokens:
            self._decisions_with_oov += 1
            self._oov_tokens.update(oov_tokens)

        decision_point = str(state.get("decision_point", ""))
        encoded = self._encode(state, decision_point).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model(encoded)
            probabilities = torch.softmax(logits.squeeze(0), dim=-1)

        allowed = set(eligible_actions)
        chosen_action: str | None = None
        chosen_score = 0.0
        for index in torch.argsort(probabilities, descending=True).tolist():
            candidate = self.inv_action.get(int(index))
            if candidate is not None and candidate in allowed:
                chosen_action = candidate
                chosen_score = float(probabilities[int(index)].item())
                break

        if chosen_action is None:
            # The deployed ppo_only stack fails the request here. Continue the
            # diagnostic simulation with a placeholder solely to enumerate
            # all problems, while marking the cell non-serving-equivalent.
            self._fallback_decisions += 1
            chosen_action = "no-op" if "no-op" in eligible_actions else eligible_actions[0]
        else:
            self._policy_scores.append(chosen_score)

        self._prior_actions.append(chosen_action)
        # Production serves deterministic argmax and therefore logs 1.0, not
        # the neural softmax score (which is diagnostic only).
        return chosen_action, 1.0

    def diagnostics(self) -> dict[str, Any]:
        return {
            "decisions": self._decisions,
            "fallback_decisions": self._fallback_decisions,
            "serving_equivalent": self._fallback_decisions == 0,
            "decisions_with_oov_tokens": self._decisions_with_oov,
            "oov_token_occurrences": self._oov_occurrences,
            "scalar_token_occurrences": self._token_occurrences,
            "oov_token_rate": (
                self._oov_occurrences / self._token_occurrences
                if self._token_occurrences else 0.0
            ),
            "oov_tokens": dict(sorted(self._oov_tokens.items())),
            "mean_selected_policy_score": _mean(self._policy_scores),
            "min_selected_policy_score": min(self._policy_scores, default=0.0),
            "production_history_recomputed": True,
        }


class InitialContextRandom(random.Random):
    """Random stream that fixes only the simulator's two initial context draws."""

    def __init__(self, seed: int, device_type: str, traffic_source: str) -> None:
        super().__init__(seed)
        self.device_type = device_type
        self.traffic_source = traffic_source
        self._forced: list[str] = []

    def begin_session(self) -> None:
        self._forced = [self.device_type, self.traffic_source]

    def choice(self, sequence):  # type: ignore[override]
        if self._forced:
            return self._forced.pop(0)
        return super().choice(sequence)


@dataclass
class CellEvaluation:
    summary: dict[str, Any]
    rewards: list[float]
    steps: list[float]
    funnel_depths: list[float]
    converted: list[bool]
    action_counts: Counter[str]


def _session_converted(transitions: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(transition.get("metadata"), dict)
        and "purchase" in (transition["metadata"].get("generated_events") or [])
        for transition in transitions
    )


FUNNEL_DEPTH = {
    "landing": 1,
    "pdp": 2,
    "cart": 3,
    "checkout": 4,
}


def _session_funnel_depth(transitions: list[dict[str, Any]]) -> float:
    """Return the preregistered deepest funnel stage for one session."""
    if not transitions:
        return 0.0
    if _session_converted(transitions):
        return 5.0
    return float(
        max(
            FUNNEL_DEPTH.get(str(transition.get("decision_point") or ""), 0)
            for transition in transitions
        )
    )


# Adjacent funnel stages, in depth order. "Reached stage k" is exactly
# "session max funnel depth >= k", so the per-stage progression proportions are
# a deterministic function of the depth distribution. The human analysis
# derives the identical quantities from its own funnel_depth field, which keeps
# the two sides of the comparison definitionally aligned: both measure the
# session-level conditional funnel advance, not the per-step Markov parameter
# in archetypes.yaml.
FUNNEL_PROGRESSION_PAIRS = (
    ("landing", "pdp"),
    ("pdp", "cart"),
    ("cart", "checkout"),
    ("checkout", "purchase"),
)


def _funnel_depth_histogram(funnel_depths: list[float]) -> dict[str, int]:
    """Count sessions at each max funnel depth (0-5)."""
    counts = Counter(int(depth) for depth in funnel_depths)
    return {str(depth): int(counts.get(depth, 0)) for depth in range(6)}


def _funnel_progression(funnel_depths: list[float]) -> dict[str, dict[str, float | int]]:
    """Conditional funnel-advance proportions per adjacent stage pair.

    For stage k (1=landing .. 5=purchase), ``reached`` counts sessions with max
    depth >= k and ``advanced`` those with max depth >= k+1, so ``proportion``
    is P(ever reach stage k+1 | ever reached stage k). Pairs whose conditioning
    set is empty report a proportion of NaN rather than a silent zero.
    """
    depths = [int(depth) for depth in funnel_depths]
    progression: dict[str, dict[str, float | int]] = {}
    for stage_index, (source, target) in enumerate(FUNNEL_PROGRESSION_PAIRS, start=1):
        reached = sum(depth >= stage_index for depth in depths)
        advanced = sum(depth >= stage_index + 1 for depth in depths)
        progression[f"{source}_to_{target}"] = {
            "reached": reached,
            "advanced": advanced,
            "proportion": advanced / reached if reached else float("nan"),
        }
    return progression


def _action_distribution(action_counts: Counter[str]) -> dict[str, dict[str, float | int]]:
    total = sum(action_counts.values())
    action_names = list(ALL_ACTIONS) + sorted(set(action_counts) - set(ALL_ACTIONS))
    return {
        action: {
            "count": int(action_counts.get(action, 0)),
            "share": action_counts.get(action, 0) / total if total else 0.0,
        }
        for action in action_names
    }


def evaluate_cell(
    policy_name: str,
    policy: Any,
    archetype: Archetype,
    n_sessions: int,
    seed: int,
    t_max: int,
    dynamics: dict[str, Any],
    device_type: str | None = None,
    traffic_source: str | None = None,
) -> CellEvaluation:
    """Evaluate one policy/archetype cell with a fresh deterministic stream."""
    if n_sessions <= 0:
        raise ValueError("n_sessions must be positive")

    rng: random.Random
    if device_type is not None and traffic_source is not None:
        rng = InitialContextRandom(seed, device_type, traffic_source)
    else:
        rng = random.Random(seed)
    simulator = SessionSimulator(dynamics=dynamics, sequential=True)
    policy.reset_diagnostics()

    rewards: list[float] = []
    steps: list[float] = []
    funnel_depths: list[float] = []
    converted: list[bool] = []
    action_counts: Counter[str] = Counter()

    for _ in range(n_sessions):
        if isinstance(rng, InitialContextRandom):
            rng.begin_session()
        policy.begin_session()
        transitions = simulator.simulate_session(
            archetype,
            policy,
            t_max=t_max,
            rng=rng,
        )
        rewards.append(sum(float(transition["reward"]) for transition in transitions))
        steps.append(float(len(transitions)))
        funnel_depths.append(_session_funnel_depth(transitions))
        converted.append(_session_converted(transitions))
        action_counts.update(str(transition["action"]) for transition in transitions)

    conversion_count = sum(converted)
    context_suffix = (
        f"__{device_type}__{traffic_source}"
        if device_type is not None and traffic_source is not None
        else ""
    )
    summary = {
        "cell_id": f"{policy_name}__{archetype.name}{context_suffix}",
        "policy": policy_name,
        "archetype": archetype.name,
        "seed": seed,
        "n_sessions": n_sessions,
        "conversion_count": conversion_count,
        "conversion_rate": conversion_count / n_sessions,
        "reward_mean": _mean(rewards),
        "reward_sd": _sample_sd(rewards),
        "total_steps": int(sum(steps)),
        "step_count_mean": _mean(steps),
        "step_count_sd": _sample_sd(steps),
        "funnel_depth_mean": _mean(funnel_depths),
        "funnel_depth_sd": _sample_sd(funnel_depths),
        "funnel_depth_histogram": _funnel_depth_histogram(funnel_depths),
        "funnel_progression": _funnel_progression(funnel_depths),
        "action_count": int(sum(action_counts.values())),
        "action_distribution": _action_distribution(action_counts),
        "diagnostics": policy.diagnostics(),
    }
    if context_suffix:
        summary["device_type"] = device_type
        summary["traffic_source"] = traffic_source
    return CellEvaluation(
        summary, rewards, steps, funnel_depths, converted, action_counts
    )


def _policy_summary(policy_name: str, cells: list[CellEvaluation]) -> dict[str, Any]:
    rewards = [value for cell in cells for value in cell.rewards]
    steps = [value for cell in cells for value in cell.steps]
    funnel_depths = [value for cell in cells for value in cell.funnel_depths]
    converted = [value for cell in cells for value in cell.converted]
    action_counts: Counter[str] = Counter()
    for cell in cells:
        action_counts.update(cell.action_counts)
    return {
        "policy": policy_name,
        "n_sessions": len(rewards),
        "conversion_count": sum(converted),
        "conversion_rate": sum(converted) / len(converted) if converted else 0.0,
        "reward_mean": _mean(rewards),
        "reward_sd": _sample_sd(rewards),
        "total_steps": int(sum(steps)),
        "step_count_mean": _mean(steps),
        "step_count_sd": _sample_sd(steps),
        "funnel_depth_mean": _mean(funnel_depths),
        "funnel_depth_sd": _sample_sd(funnel_depths),
        "funnel_depth_histogram": _funnel_depth_histogram(funnel_depths),
        "funnel_progression": _funnel_progression(funnel_depths),
        "action_count": int(sum(action_counts.values())),
        "action_distribution": _action_distribution(action_counts),
    }


def _bandit_metadata(db_path: Path) -> dict[str, Any]:
    with sqlite3.connect(str(db_path)) as connection:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='bandit_arm_stats'"
        ).fetchone()
        if table is None:
            raise ValueError(f"bandit_arm_stats table not found in {db_path}")
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(bandit_arm_stats)")
        }
        row_count, context_count = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT decision_point || '|' || context_key) "
            "FROM bandit_arm_stats"
        ).fetchone()
        schema_versions: list[int] = []
        if "schema_version" in columns:
            schema_versions = [
                int(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT schema_version FROM bandit_arm_stats "
                    "WHERE schema_version IS NOT NULL ORDER BY schema_version"
                )
            ]
    return {
        "arm_rows": int(row_count),
        "context_count": int(context_count),
        "schema_versions": schema_versions,
        "serving_mode": "frozen_deterministic_greedy",
        "prior_count": PRIOR_COUNT,
        "prior_mean": PRIOR_MEAN,
    }


def _ppo_metadata(policy: FrozenPPOPolicy) -> dict[str, Any]:
    hidden_sizes = [
        int(module.out_features)
        for module in policy.model.backbone
        if hasattr(module, "out_features")
    ]
    return {
        "state_vocab_size": len(policy.state_vocab),
        "action_vocab_size": len(policy.action_vocab),
        "action_vocab": dict(sorted(policy.action_vocab.items())),
        "hidden_sizes": hidden_sizes,
        "serving_mode": "frozen_deterministic_masked_argmax",
        "production_history_recomputed": True,
    }


def build_payload(
    study_id: str,
    bandit_db: Path,
    ppo_checkpoint: Path,
    archetypes_path: Path,
    sessions_per_cell: int,
    base_seed: int,
    t_max_override: int | None = None,
    context_sessions_per_stratum: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run ten aggregate cells plus optional exogenous-context strata."""
    if torch is None:
        raise RuntimeError("PyTorch is required to evaluate the V3 PPO checkpoint")
    if sessions_per_cell <= 0:
        raise ValueError("sessions_per_cell must be positive")
    if context_sessions_per_stratum < 0:
        raise ValueError("context_sessions_per_stratum must be non-negative")
    for path, label in (
        (bandit_db, "V2 bandit database"),
        (ppo_checkpoint, "V3 PPO checkpoint"),
        (archetypes_path, "archetype config"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    # PPO inference is deterministic; pinning the CPU thread count also keeps
    # the runtime setup explicit in the manifest.
    torch.set_num_threads(1)
    torch.manual_seed(base_seed)

    archetypes, mixture_prior, full_config = load_archetypes(str(archetypes_path))
    archetype_names = list(archetypes)
    if not archetype_names:
        raise ValueError("archetype config contains no archetypes")
    seeds = derive_archetype_seeds(archetype_names, base_seed)
    dynamics = merge_dynamics(full_config.get("dynamics", {}) or {})
    config_t_max = int(full_config.get("simulation", {}).get("t_max", 20))
    t_max = int(t_max_override) if t_max_override is not None else config_t_max
    if t_max <= 0:
        raise ValueError("t_max must be positive")

    policies: dict[str, Any] = {
        "v2_bandit": FrozenBanditPolicy(bandit_db),
        "v3_ppo": FrozenPPOPolicy(ppo_checkpoint),
    }
    evaluated: dict[str, list[CellEvaluation]] = {name: [] for name in policies}

    for policy_name, policy in policies.items():
        for archetype_name in archetype_names:
            print(
                f"[baseline] {policy_name:<10} x {archetype_name:<20} "
                f"n={sessions_per_cell} seed={seeds[archetype_name]}"
            )
            evaluated[policy_name].append(
                evaluate_cell(
                    policy_name=policy_name,
                    policy=policy,
                    archetype=archetypes[archetype_name],
                    n_sessions=sessions_per_cell,
                    seed=seeds[archetype_name],
                    t_max=t_max,
                    dynamics=dynamics,
                )
            )

    context_cells: list[dict[str, Any]] = []
    if context_sessions_per_stratum:
        for policy_index, (policy_name, policy) in enumerate(policies.items()):
            for archetype_index, archetype_name in enumerate(archetype_names):
                for device_index, device_type in enumerate(DEVICE_STRATA):
                    for traffic_index, traffic_source in enumerate(TRAFFIC_STRATA):
                        context_seed = derive_context_seed(
                            policy_index,
                            archetype_index,
                            device_index,
                            traffic_index,
                            base_seed,
                        )
                        print(
                            f"[context] {policy_name:<10} x {archetype_name:<20} "
                            f"x {device_type:<7} x {traffic_source:<8} "
                            f"n={context_sessions_per_stratum} seed={context_seed}"
                        )
                        context_cells.append(
                            evaluate_cell(
                                policy_name=policy_name,
                                policy=policy,
                                archetype=archetypes[archetype_name],
                                n_sessions=context_sessions_per_stratum,
                                seed=context_seed,
                                t_max=t_max,
                                dynamics=dynamics,
                                device_type=device_type,
                                traffic_source=traffic_source,
                            ).summary
                        )

    cells = [cell.summary for name in policies for cell in evaluated[name]]
    code_inputs = [
        Path(__file__),
        REPO_ROOT / "CustomerSimulation" / "simulation" / "simulator.py",
        REPO_ROOT / "CustomerSimulation" / "simulation" / "state.py",
        REPO_ROOT / "SharedSchema" / "shared_schema" / "constants.py",
        REPO_ROOT / "SharedSchema" / "shared_schema" / "features.py",
    ]
    payload: dict[str, Any] = {
        "schema_version": 2,
        "study_id": study_id,
        "purpose": "policy-matched pre-recruitment simulator reference",
        "git_head": _git_head(),
        "design": {
            "policy_levels": list(policies),
            "archetype_levels": archetype_names,
            "n_cells": len(cells),
            "sessions_per_cell": sessions_per_cell,
            "aggregate_sessions": len(cells) * sessions_per_cell,
            "total_sessions": (
                len(cells) * sessions_per_cell
                + len(context_cells) * context_sessions_per_stratum
            ),
            "equal_allocation": True,
            "paired_seed_within_archetype": True,
            "context_standardization": {
                "enabled": bool(context_cells),
                "device_levels": list(DEVICE_STRATA),
                "traffic_levels": list(TRAFFIC_STRATA),
                "sessions_per_stratum": context_sessions_per_stratum,
                "n_strata_cells": len(context_cells),
                "additional_sessions": len(context_cells) * context_sessions_per_stratum,
                "independent_seeds_across_strata_cells": True,
                "rule": (
                    "Primary fidelity standardizes simulator cell means to the "
                    "recorded joint device/traffic distribution within each policy-persona cell."
                ),
            },
        },
        "simulation": {
            "base_seed": base_seed,
            "archetype_seeds": seeds,
            "t_max": t_max,
            "sequential_state": True,
            "dynamics": dynamics,
            "configured_mixture_prior_not_used_for_equal_cells": mixture_prior,
        },
        "inputs": {
            "archetype_config": _input_descriptor(archetypes_path),
            "v2_bandit_db": {
                **_input_descriptor(bandit_db),
                **_bandit_metadata(bandit_db),
            },
            "v3_ppo_checkpoint": {
                **_input_descriptor(ppo_checkpoint),
                **_ppo_metadata(policies["v3_ppo"]),
            },
            "code": [_input_descriptor(path) for path in code_inputs],
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_num_threads": torch.get_num_threads(),
        },
        "policy_summaries_equal_archetype_weight": {
            name: _policy_summary(name, evaluated[name]) for name in policies
        },
        "cells": cells,
        "context_strata": context_cells,
        "interpretation_boundary": (
            "These are simulator-oracle predictions for two complete frozen policy "
            "systems; they do not identify a sequential mechanism or online learning effect."
        ),
    }
    return payload, cells


def _csv_rows(study_id: str, cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        row: dict[str, Any] = {
            "study_id": study_id,
            "cell_id": cell["cell_id"],
            "policy": cell["policy"],
            "archetype": cell["archetype"],
            "seed": cell["seed"],
            "n_sessions": cell["n_sessions"],
            "conversion_count": cell["conversion_count"],
            "conversion_rate": cell["conversion_rate"],
            "reward_mean": cell["reward_mean"],
            "reward_sd": cell["reward_sd"],
            "total_steps": cell["total_steps"],
            "step_count_mean": cell["step_count_mean"],
            "step_count_sd": cell["step_count_sd"],
            "funnel_depth_mean": cell["funnel_depth_mean"],
            "funnel_depth_sd": cell["funnel_depth_sd"],
            "action_count": cell["action_count"],
        }
        for depth, count in cell["funnel_depth_histogram"].items():
            row[f"funnel_depth_{depth}_count"] = count
        for pair, values in cell["funnel_progression"].items():
            row[f"progression_{pair}_reached"] = values["reached"]
            row[f"progression_{pair}_advanced"] = values["advanced"]
            row[f"progression_{pair}_proportion"] = values["proportion"]
        for action, values in cell["action_distribution"].items():
            column_action = action.replace("-", "_")
            row[f"action_{column_action}_count"] = values["count"]
            row[f"action_{column_action}_share"] = values["share"]
        diagnostics = cell["diagnostics"]
        for name in (
            "fallback_decisions",
            "missing_feature_decisions",
            "unseen_context_decisions",
            "unseen_context_rate",
            "partially_unseen_context_decisions",
            "selected_unseen_arm_decisions",
            "decisions_with_oov_tokens",
            "oov_token_occurrences",
            "scalar_token_occurrences",
            "oov_token_rate",
            "mean_selected_policy_score",
        ):
            row[name] = diagnostics.get(name, 0)
        row["diagnostics_json"] = json.dumps(diagnostics, sort_keys=True, separators=(",", ":"))
        rows.append(row)
    return rows


def write_outputs(
    output_dir: Path,
    payload: dict[str, Any],
    cells: list[dict[str, Any]],
) -> tuple[Path, Path]:
    """Write a new artifact pair and refuse every non-empty target directory."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; choose a new --study-id or --out-dir"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / JSON_NAME
    csv_path = output_dir / CSV_NAME

    rows = _csv_rows(str(payload["study_id"]), cells)
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    fieldnames = list(rows[0]) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--study-id", default=DEFAULT_STUDY_ID)
    parser.add_argument("--sessions-per-cell", type=int, default=2000)
    parser.add_argument(
        "--context-sessions-per-stratum",
        type=int,
        default=500,
        help=(
            "Sessions for each policy x archetype x device x traffic stratum; "
            "set 0 only for a non-release diagnostic"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--bandit-db",
        type=Path,
        default=REPO_ROOT / "DemoSiteV2" / "data" / "demosite.db",
    )
    parser.add_argument(
        "--ppo-checkpoint",
        type=Path,
        default=REPO_ROOT / "OfflineTraining" / "outputs" / "ppo_policy.pt",
    )
    parser.add_argument(
        "--archetypes",
        type=Path,
        default=REPO_ROOT / "CustomerSimulation" / "config" / "archetypes.yaml",
    )
    parser.add_argument(
        "--t-max",
        type=int,
        default=None,
        help="Override simulation.t_max (recorded in the manifest).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: Experiments/study_policy_baselines/<study-id>).",
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.study_id):
        parser.error("--study-id may contain only letters, digits, '.', '_' and '-'")
    if args.sessions_per_cell <= 0:
        parser.error("--sessions-per-cell must be positive")
    if args.context_sessions_per_stratum < 0:
        parser.error("--context-sessions-per-stratum must be non-negative")
    if args.t_max is not None and args.t_max <= 0:
        parser.error("--t-max must be positive")
    if args.out_dir is None:
        args.out_dir = REPO_ROOT / "Experiments" / "study_policy_baselines" / args.study_id
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload, cells = build_payload(
        study_id=args.study_id,
        bandit_db=args.bandit_db.resolve(),
        ppo_checkpoint=args.ppo_checkpoint.resolve(),
        archetypes_path=args.archetypes.resolve(),
        sessions_per_cell=args.sessions_per_cell,
        base_seed=args.seed,
        t_max_override=args.t_max,
        context_sessions_per_stratum=args.context_sessions_per_stratum,
    )
    json_path, csv_path = write_outputs(args.out_dir.resolve(), payload, cells)

    print("\nPolicy-matched archetype baseline")
    print(f"{'POLICY':<12}{'ARCHETYPE':<22}{'CONV':>9}{'REWARD':>11}{'SD':>9}{'STEPS':>9}")
    print("-" * 72)
    for cell in cells:
        print(
            f"{cell['policy']:<12}{cell['archetype']:<22}"
            f"{cell['conversion_rate']:>9.3f}{cell['reward_mean']:>11.3f}"
            f"{cell['reward_sd']:>9.3f}{cell['step_count_mean']:>9.2f}"
        )
    print(f"\nJSON: {json_path}")
    print(f"CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
