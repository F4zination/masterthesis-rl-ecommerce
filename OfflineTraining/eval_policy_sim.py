#!/usr/bin/env python3
"""Simulator-based policy evaluation.

Rolls out trained_policy.json, a uniform-random baseline, and a no-op
baseline through the CustomerSimulation engine and compares per-session
rewards head-to-head.

Unlike OPE (which relies on importance-sampling approximations over logged
data), this gives exact on-policy returns — at the cost of requiring the
simulator to accurately reflect production dynamics.

Usage::

    python eval_policy_sim.py --policy-file outputs/trained_policy.json
    python eval_policy_sim.py \\
        --policy-file outputs/trained_policy.json \\
        --archetypes ../CustomerSimulation/config/archetypes.yaml \\
        --n-sessions 2000 --seed 7 --temperature 0.0
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Resolve repo root so CustomerSimulation and SharedSchema are importable
# without requiring a full editable install.
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "CustomerSimulation"))
sys.path.insert(0, str(_repo_root / "SharedSchema"))

from simulation.archetype import Archetype, load_archetypes  # noqa: E402
from simulation.behavior_policy import BehaviorPolicy  # noqa: E402
from simulation.simulator import SessionSimulator  # noqa: E402


# ---------------------------------------------------------------------------
# Policy implementations
# ---------------------------------------------------------------------------

class NoOpPolicy(BehaviorPolicy):
    """Always selects 'no-op'; falls back to first eligible action if absent."""

    def select_action(
        self,
        eligible_actions: list[str],
        archetype_name: str | None = None,
        rng: random.Random | None = None,
    ) -> tuple[str, float]:
        if "no-op" in eligible_actions:
            return "no-op", 1.0
        return eligible_actions[0], 1.0


class TrainedPolicy(BehaviorPolicy):
    """Wraps a trained_policy.json to match the BehaviorPolicy interface.

    Because ``SessionSimulator.simulate_session()`` does not pass the state
    dict to ``select_action()``, this class exposes a ``set_state()`` hook
    that must be called before each ``select_action()`` invocation.
    ``StateAwareSimulator`` handles this transparently.

    When ``temperature > 0`` the policy samples actions proportionally to
    ``softmax(Q(s, ·) / T)``; temperature=0 (default) is greedy/deterministic.
    """

    def __init__(self, policy_path: Path | str, temperature: float = 0.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.current_state: dict[str, Any] = {}

        payload = json.loads(Path(policy_path).read_text(encoding="utf-8"))
        self._explicit: dict[str, str] = payload.get("policy") or {}
        self._default_by_point: dict[str, str] = (
            payload.get("default_action_by_decision_point") or {}
        )

        # Parse Q-table: keys are "state_json|||action"
        self._q_by_state: dict[str, dict[str, float]] = defaultdict(dict)
        for key, qval in (payload.get("q_table") or {}).items():
            parts = key.split("|||", 1)
            if len(parts) == 2:
                self._q_by_state[parts[0]][parts[1]] = float(qval)

    def set_state(self, state_dict: dict[str, Any]) -> None:
        """Inject current state before select_action() is called."""
        self.current_state = state_dict

    def select_action(
        self,
        eligible_actions: list[str],
        archetype_name: str | None = None,
        rng: random.Random | None = None,
    ) -> tuple[str, float]:
        rand = rng or random

        sk = json.dumps(self.current_state, sort_keys=True, ensure_ascii=True)
        point = str(self.current_state.get("decision_point", ""))

        if self.temperature <= 0.0 or not self._q_by_state.get(sk):
            # Greedy / deterministic fallback
            target = (
                self._explicit.get(sk)
                or self._default_by_point.get(point)
                or "no-op"
            )
            if target not in eligible_actions:
                target = "no-op" if "no-op" in eligible_actions else eligible_actions[0]
            # Return uniform propensity so logs remain valid for OPE if reused
            return target, 1.0 / len(eligible_actions)

        # Softmax sampling
        q_vals = self._q_by_state[sk]
        scores = [q_vals.get(a, 0.0) / self.temperature for a in eligible_actions]
        max_s = max(scores)
        exps = [math.exp(s - max_s) for s in scores]
        total = sum(exps)
        probs = [e / total for e in exps]
        target = rand.choices(eligible_actions, weights=probs, k=1)[0]
        propensity = probs[eligible_actions.index(target)]
        return target, propensity


# ---------------------------------------------------------------------------
# State-aware simulator
# ---------------------------------------------------------------------------

class StateAwareSimulator(SessionSimulator):
    """Thin alias over the shared :class:`SessionSimulator`.

    The base simulator now injects ``policy.set_state(...)`` before every
    ``select_action()`` (when the policy supports the hook), writes
    ``metadata.generated_events`` so the eval can count *actual* purchases, and
    honours the ``dynamics``/``sequential`` configuration passed to its
    constructor.  This subclass is retained only so existing imports keep
    working; it adds no behaviour of its own.
    """


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _mean(xs: list[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def _std(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def build_dynamics_from_args(full_config: dict, args: Any) -> dict:
    """Merge the config ``dynamics`` block with per-knob CLI overrides.

    Shared by ``eval_policy_sim`` and ``eval_all_policies_sim`` so both eval
    entry points resolve the environment dynamics identically.  Returns a plain
    dict (the simulator normalises it over ``DEFAULT_DYNAMICS``).
    """
    dynamics: dict = dict(full_config.get("dynamics", {}) or {})
    for key in (
        "fatigue_rate",
        "fatigue_window",
        "delayed_reward_strength",
        "delayed_reward_decay",
        "transition_coupling_strength",
    ):
        val = getattr(args, key, None)
        if val is not None:
            dynamics[key] = val
    return dynamics


def run_policy(
    policy: BehaviorPolicy,
    archetypes: dict[str, Archetype],
    mixture_prior: dict[str, float],
    n_sessions: int,
    rng: random.Random,
    t_max: int = 20,
    dynamics: dict[str, Any] | None = None,
    sequential: bool = False,
) -> dict[str, Any]:
    """Run n_sessions and return aggregate statistics.

    ``dynamics`` and ``sequential`` are forwarded to the simulator so the eval
    rolls out under the *same* environment that produced the training data
    (the fairness invariant — both policies see identical dynamics/reward).
    """
    sim = StateAwareSimulator(dynamics=dynamics, sequential=sequential)
    arch_names = list(mixture_prior.keys())
    arch_weights = [mixture_prior[a] for a in arch_names]

    session_rewards: list[float] = []
    session_steps: list[float] = []
    purchase_count = 0
    purchase_count_proxy = 0
    have_events = False
    action_counts: dict[str, int] = defaultdict(int)

    for _ in range(n_sessions):
        archetype_name = rng.choices(arch_names, weights=arch_weights, k=1)[0]
        archetype = archetypes[archetype_name]
        transitions = sim.simulate_session(archetype, policy, t_max=t_max, rng=rng)

        total_reward = sum(t["reward"] for t in transitions)
        session_rewards.append(total_reward)
        session_steps.append(len(transitions))

        session_purchased = False
        for tr in transitions:
            action_counts[tr["action"]] += 1
            md = tr.get("metadata")
            if isinstance(md, dict) and "generated_events" in md:
                have_events = True
                if "purchase" in (md.get("generated_events") or []):
                    session_purchased = True
        if session_purchased:
            purchase_count += 1

        # Reward-threshold proxy retained as a labelled fallback (and a
        # cross-check when delayed reward makes the threshold noisier).
        if total_reward > 0.5:
            purchase_count_proxy += 1

    # Prefer the exact event-based count; fall back to the proxy if the
    # simulator did not emit metadata for some reason.
    exact_rate = purchase_count / max(n_sessions, 1)
    proxy_rate = purchase_count_proxy / max(n_sessions, 1)
    return {
        "n_sessions": n_sessions,
        "mean_reward_per_session": _mean(session_rewards),
        "std_reward_per_session": _std(session_rewards),
        "mean_steps_per_session": _mean(session_steps),
        "purchase_rate": exact_rate if have_events else proxy_rate,
        "purchase_rate_exact": exact_rate if have_events else None,
        "purchase_rate_approx": proxy_rate,
        "action_distribution": dict(action_counts),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _default_archetypes = str(
        _repo_root / "CustomerSimulation" / "config" / "archetypes.yaml"
    )
    parser = argparse.ArgumentParser(
        description="Evaluate trained policy via CustomerSimulation oracle"
    )
    parser.add_argument(
        "--policy-file",
        required=True,
        help="trained_policy.json produced by train_offline_policy.py",
    )
    parser.add_argument(
        "--archetypes",
        default=_default_archetypes,
        help="Path to archetypes.yaml (default: CustomerSimulation/config/archetypes.yaml)",
    )
    parser.add_argument(
        "--n-sessions",
        type=int,
        default=2000,
        help="Sessions to run per policy (default: 2000)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Softmax temperature for TrainedPolicy (0=greedy, default: 0.0)",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "outputs"),
        help="Directory for output artifacts",
    )
    parser.add_argument(
        "--t-max",
        type=int,
        default=20,
        help="Maximum steps per simulated session (default: 20)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Emit trajectory-history features (read by sequential policies, "
             "ignored by the bandit). Match this to how the dataset was generated.",
    )
    parser.add_argument("--fatigue-rate", type=float, default=None)
    parser.add_argument("--fatigue-window", type=int, default=None)
    parser.add_argument("--delayed-reward-strength", type=float, default=None)
    parser.add_argument("--delayed-reward-decay", type=float, default=None)
    parser.add_argument("--transition-coupling-strength", type=float, default=None)
    args = parser.parse_args()

    policy_path = Path(args.policy_file)
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    archetypes_path = Path(args.archetypes)
    if not archetypes_path.exists():
        raise FileNotFoundError(f"Archetypes config not found: {archetypes_path}")

    print(f"[sim_eval] Loading archetypes from {archetypes_path}")
    archetypes, mixture_prior, full_config = load_archetypes(str(archetypes_path))
    print(f"[sim_eval] Archetypes: {list(archetypes.keys())}")

    dynamics = build_dynamics_from_args(full_config, args)
    print(f"[sim_eval] sequential={args.sequential}  dynamics_overrides="
          f"{ {k: dynamics[k] for k in ('fatigue_rate', 'delayed_reward_strength', 'transition_coupling_strength')} }")

    policies: dict[str, BehaviorPolicy] = {
        "trained_policy": TrainedPolicy(policy_path, temperature=args.temperature),
        "uniform_random": BehaviorPolicy(),
        "no_op": NoOpPolicy(),
    }

    results: dict[str, Any] = {
        "policy_file": str(policy_path),
        "archetypes": str(archetypes_path),
        "n_sessions": args.n_sessions,
        "seed": args.seed,
        "temperature": args.temperature,
        "policies": {},
    }

    for name, policy in policies.items():
        rng = random.Random(args.seed)
        print(f"[sim_eval] Running '{name}' for {args.n_sessions:,} sessions...")
        stats = run_policy(
            policy,
            archetypes,
            mixture_prior,
            n_sessions=args.n_sessions,
            rng=rng,
            t_max=args.t_max,
            dynamics=dynamics,
            sequential=args.sequential,
        )
        results["policies"][name] = stats
        print(
            f"[sim_eval]   '{name}'  mean_reward={stats['mean_reward_per_session']:.4f}"
            f"  ±{stats['std_reward_per_session']:.4f}"
            f"  purchase_rate={stats['purchase_rate']:.3f}"
        )

    # Normalized lift relative to uniform baseline
    pr = results["policies"]
    if "trained_policy" in pr and "uniform_random" in pr and "no_op" in pr:
        v_trained = pr["trained_policy"]["mean_reward_per_session"]
        v_uniform = pr["uniform_random"]["mean_reward_per_session"]
        v_noop = pr["no_op"]["mean_reward_per_session"]
        denom = v_uniform - v_noop
        lift = (v_trained - v_noop) / denom if abs(denom) > 1e-9 else float("nan")
        results["lift_vs_noop_normalized"] = lift
        print(f"\n[sim_eval] Normalized lift (reward): {lift:.4f}  (>1.0 = beats uniform baseline)")

    # Comparison table
    print("\n{'Policy':<22} {'Mean reward/session':>22} {'Std':>8} {'Purchase rate':>15}")
    print("-" * 70)
    for name, stats in pr.items():
        print(
            f"{name:<22} {stats['mean_reward_per_session']:>22.4f}"
            f" {stats['std_reward_per_session']:>8.4f}"
            f" {stats['purchase_rate']:>15.3f}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sim_eval_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\n[sim_eval] Results written to {out_path}")


if __name__ == "__main__":
    main()
