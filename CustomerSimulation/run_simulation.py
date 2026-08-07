#!/usr/bin/env python3
"""Generate simulated customer journey data for offline RL training.

Usage:
    python run_simulation.py --n-sessions 5000 --seed 42
    python run_simulation.py --output-dir ../OfflineTraining/outputs/simulated/
"""
import argparse
import random
import sys
from pathlib import Path

# Make SharedSchema importable without installation when running from this dir.
_repo_root = Path(__file__).resolve().parent.parent
_shared_schema_path = _repo_root / "SharedSchema"
if str(_shared_schema_path) not in sys.path:
    sys.path.insert(0, str(_shared_schema_path))

from simulation.archetype import load_archetypes
from simulation.behavior_policy import BehaviorPolicy
from simulation.simulator import SessionSimulator
from simulation.writer import DatasetWriter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate simulated customer journey transitions for RL training."
    )
    parser.add_argument("--n-sessions", type=int, default=1000, metavar="N",
                        help="Number of sessions to simulate (default: 1000).")
    parser.add_argument("--config", default="config/archetypes.yaml", metavar="PATH",
                        help="Path to archetypes YAML config (default: config/archetypes.yaml).")
    parser.add_argument("--output-dir", default="output", metavar="DIR",
                        help="Directory for output files (default: output/).")
    parser.add_argument("--epsilon", type=float, default=None, metavar="E",
                        help="Override behavior policy epsilon from config.")
    parser.add_argument("--seed", type=int, default=None, metavar="S",
                        help="Random seed for reproducibility.")
    parser.add_argument("--t-max", type=int, default=None, metavar="T",
                        help="Override simulation.t_max horizon from config.")
    parser.add_argument("--sequential", action="store_true",
                        help="Emit bucketed trajectory-history features in the state "
                             "dict (read by PPO, ignored by the bandit).")
    # Per-knob dynamics overrides (default None → fall back to config / off).
    parser.add_argument("--fatigue-rate", type=float, default=None)
    parser.add_argument("--fatigue-window", type=int, default=None)
    parser.add_argument("--delayed-reward-strength", type=float, default=None)
    parser.add_argument("--delayed-reward-decay", type=float, default=None)
    parser.add_argument("--transition-coupling-strength", type=float, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    archetypes, mixture_prior, full_config = load_archetypes(args.config)

    config_t_max: int = full_config.get("simulation", {}).get("t_max", 20)
    t_max: int = args.t_max if args.t_max is not None else config_t_max
    bp_config: dict = full_config.get("behavior_policy", {})
    epsilon: float = args.epsilon if args.epsilon is not None else bp_config.get("epsilon", 0.2)
    per_archetype_epsilon: dict = bp_config.get("per_archetype_epsilon", {})

    # Dynamics: start from the config block, then apply any CLI overrides.
    dynamics: dict = dict(full_config.get("dynamics", {}) or {})
    _knob_overrides = {
        "fatigue_rate": args.fatigue_rate,
        "fatigue_window": args.fatigue_window,
        "delayed_reward_strength": args.delayed_reward_strength,
        "delayed_reward_decay": args.delayed_reward_decay,
        "transition_coupling_strength": args.transition_coupling_strength,
    }
    for _key, _val in _knob_overrides.items():
        if _val is not None:
            dynamics[_key] = _val

    policy = BehaviorPolicy(epsilon=epsilon, per_archetype_epsilon=per_archetype_epsilon)
    simulator = SessionSimulator(dynamics=dynamics, sequential=args.sequential)
    writer = DatasetWriter()

    archetype_names = list(mixture_prior.keys())
    archetype_weights = [mixture_prior[n] for n in archetype_names]
    archetype_counts: dict[str, int] = {name: 0 for name in archetype_names}

    all_sessions: list[list[dict]] = []
    _active_knobs = {k: dynamics[k] for k in (
        "fatigue_rate", "delayed_reward_strength", "transition_coupling_strength"
    ) if dynamics.get(k)}
    print(f"Simulating {args.n_sessions} sessions  (t_max={t_max}, epsilon={epsilon}, seed={args.seed}, "
          f"sequential={args.sequential}, active_knobs={_active_knobs or 'none'})")

    for i in range(args.n_sessions):
        archetype_name = random.choices(archetype_names, weights=archetype_weights, k=1)[0]
        archetype = archetypes[archetype_name]
        archetype_counts[archetype_name] += 1
        session = simulator.simulate_session(archetype, policy, t_max=t_max)
        all_sessions.append(session)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{args.n_sessions}")

    jsonl_path, csv_path, summary_path = writer.write(all_sessions, args.output_dir, archetype_counts)

    n_transitions = sum(len(s) for s in all_sessions)
    print(f"\nDone.  {args.n_sessions} sessions  ->  {n_transitions} transitions")
    print(f"  JSONL:   {jsonl_path}")
    print(f"  CSV:     {csv_path}")
    print(f"  Summary: {summary_path}")
    print("\nArchetype distribution:")
    for name in archetype_names:
        count = archetype_counts[name]
        print(f"  {name:<22} {count:>5}  ({100 * count / args.n_sessions:.1f}%)")


if __name__ == "__main__":
    main()
