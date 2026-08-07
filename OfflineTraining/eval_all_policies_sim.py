#!/usr/bin/env python3
"""Compare every *served* policy through the CustomerSimulation oracle.

OPE (``ope_eval.py``) and ``eval_policy_sim.py`` can only score
``trained_policy.json``. But the frozen study actually ships two *different*
served policies — DemoSiteV3's PPO checkpoint and DemoSiteV2's epsilon-greedy
bandit (stored as ``bandit_arm_stats``) — neither of which those tools can load.

This script rolls out all of them through the *same* simulator and seed, so the
mean reward per session is an apples-to-apples prediction of the frozen
V2-vs-V3 A/B *before* recruiting participants:

  * ``ppo``            - DemoSiteV3 served policy (``ppo_policy.pt``)
  * ``bandit``         - DemoSiteV2 served policy (greedy over ``demosite.db``)
  * ``tabular``        - DemoSiteV3 offline fallback (``trained_policy.json``)
  * ``uniform_random`` - random baseline
  * ``no_op``          - never intervene (control floor)

The bandit is evaluated *greedily* (its learned intent). The live site adds
EPSILON=0.2 exploration, which would pull realized reward slightly toward the
uniform baseline.

Usage::

    python eval_all_policies_sim.py \\
        --ppo outputs/ppo_policy.pt \\
        --bandit-db ../DemoSiteV2/data/demosite.db \\
        --tabular outputs/trained_policy.json \\
        --n-sessions 4000 --seed 123 --out-dir outputs/
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "OfflineTraining"))
sys.path.insert(0, str(_repo_root / "CustomerSimulation"))
sys.path.insert(0, str(_repo_root / "SharedSchema"))

# Reuse the simulator harness + baseline/tabular policies from eval_policy_sim.
from eval_policy_sim import (  # noqa: E402
    BehaviorPolicy, NoOpPolicy, TrainedPolicy, run_policy, load_archetypes,
    build_dynamics_from_args,
)
from shared_schema.features import POINT_FEATURES  # noqa: E402
from shared_schema.constants import PRIOR_COUNT, PRIOR_MEAN  # noqa: E402

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


# ── PPO (mirrors DemoSiteV3/app/services/ppo.py inference exactly) ───────────
if torch is not None and nn is not None:

    class _ActorCritic(nn.Module):
        def __init__(self, input_dim: int, action_dim: int, hidden_sizes: tuple[int, ...]) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            last = input_dim
            for h in hidden_sizes:
                layers.append(nn.Linear(last, h))
                layers.append(nn.Tanh())
                last = h
            self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
            self.policy_head = nn.Linear(last, action_dim)
            self.value_head = nn.Linear(last, 1)

        def forward(self, x):
            f = self.backbone(x)
            return self.policy_head(f), self.value_head(f).squeeze(-1)


class PPOPolicy(BehaviorPolicy):
    """DemoSiteV3's served PPO policy, loaded from the checkpoint."""

    def __init__(self, ckpt_path: str | Path) -> None:
        super().__init__()
        if torch is None:
            raise RuntimeError("torch is required to evaluate the PPO policy")
        payload = torch.load(ckpt_path, map_location="cpu")
        self.state_vocab: dict[str, int] = payload["state_vocab"]
        self.action_vocab: dict[str, int] = payload["action_vocab"]
        self.inv_action = {int(i): a for a, i in self.action_vocab.items()}
        hidden = tuple(int(x) for x in (payload.get("hidden_sizes") or (128, 64)))
        self.model = _ActorCritic(len(self.state_vocab), len(self.action_vocab), hidden)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self.current_state: dict[str, Any] = {}

    def set_state(self, state_dict: dict[str, Any]) -> None:
        self.current_state = state_dict

    def _encode(self, state: dict[str, Any], dp: str):
        vec = torch.zeros(len(self.state_vocab), dtype=torch.float32)
        for key, value in sorted(state.items()):
            if value is None or isinstance(value, (dict, list)):
                continue
            idx = self.state_vocab.get(f"{key}={value}")
            if idx is not None:
                vec[idx] = 1.0
        idx = self.state_vocab.get(f"decision_point={dp}")
        if idx is not None:
            vec[idx] = 1.0
        return vec

    def select_action(self, eligible_actions, archetype_name=None, rng=None):
        dp = str(self.current_state.get("decision_point", ""))
        x = self._encode(self.current_state, dp).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model(x)
            probs = torch.softmax(logits.squeeze(0), dim=-1)
        allowed = set(eligible_actions)
        for i in torch.argsort(probs, descending=True).tolist():
            cand = self.inv_action.get(int(i))
            if cand is not None and cand in allowed:
                return cand, max(float(probs[int(i)].item()), 1e-8)
        fallback = "no-op" if "no-op" in eligible_actions else eligible_actions[0]
        return fallback, 1.0 / len(eligible_actions)


# ── Bandit (mirrors DemoSiteV2/app/services/decision.py greedy scoring) ──────
class BanditPolicy(BehaviorPolicy):
    """DemoSiteV2's served bandit (greedy over frozen ``bandit_arm_stats``)."""

    def __init__(self, db_path: str | Path) -> None:
        super().__init__()
        conn = sqlite3.connect(str(db_path))
        self.arms: dict[tuple[str, str, str], tuple[int, float]] = {}
        for dp, ck, action, imp, rs in conn.execute(
            "SELECT decision_point, context_key, action, impressions, reward_sum FROM bandit_arm_stats"
        ):
            self.arms[(dp, ck, action)] = (int(imp), float(rs))
        conn.close()
        self.current_state: dict[str, Any] = {}

    def set_state(self, state_dict: dict[str, Any]) -> None:
        self.current_state = state_dict

    def _score(self, dp: str, ck: str, action: str) -> float:
        imp, rs = self.arms.get((dp, ck, action), (0, 0.0))
        return (rs + PRIOR_COUNT * PRIOR_MEAN) / (imp + PRIOR_COUNT)

    def select_action(self, eligible_actions, archetype_name=None, rng=None):
        s = self.current_state
        dp = str(s.get("decision_point", ""))
        feats = POINT_FEATURES.get(dp)
        if feats is None or any(f not in s for f in feats):
            fallback = "no-op" if "no-op" in eligible_actions else eligible_actions[0]
            return fallback, 1.0 / len(eligible_actions)
        ck = "|".join([dp] + [str(s[f]) for f in feats])
        best_a, best_score = eligible_actions[0], None
        for a in eligible_actions:
            sc = self._score(dp, ck, a)
            if best_score is None or sc > best_score:
                best_score, best_a = sc, a
        return best_a, 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ppo", help="ppo_policy.pt (DemoSiteV3 served policy)")
    ap.add_argument("--bandit-db", help="DemoSiteV2 demosite.db with bandit_arm_stats")
    ap.add_argument("--tabular", help="trained_policy.json (DemoSiteV3 offline fallback)")
    ap.add_argument("--archetypes", default=str(_repo_root / "CustomerSimulation" / "config" / "archetypes.yaml"))
    ap.add_argument("--n-sessions", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--t-max", type=int, default=20)
    ap.add_argument("--sequential", action="store_true",
                    help="Emit trajectory-history features (PPO conditions on them; "
                         "the bandit ignores them). Match how the policies were trained.")
    ap.add_argument("--fatigue-rate", type=float, default=None)
    ap.add_argument("--fatigue-window", type=int, default=None)
    ap.add_argument("--delayed-reward-strength", type=float, default=None)
    ap.add_argument("--delayed-reward-decay", type=float, default=None)
    ap.add_argument("--transition-coupling-strength", type=float, default=None)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "outputs"))
    args = ap.parse_args()

    archetypes, mixture_prior, full_config = load_archetypes(args.archetypes)
    dynamics = build_dynamics_from_args(full_config, args)

    policies: dict[str, BehaviorPolicy] = {}
    if args.ppo and Path(args.ppo).exists():
        policies["ppo (V3)"] = PPOPolicy(args.ppo)
    if args.bandit_db and Path(args.bandit_db).exists():
        policies["bandit (V2)"] = BanditPolicy(args.bandit_db)
    if args.tabular and Path(args.tabular).exists():
        policies["tabular (V3 fallback)"] = TrainedPolicy(args.tabular, temperature=0.0)
    policies["uniform_random"] = BehaviorPolicy()
    policies["no_op"] = NoOpPolicy()

    results: dict[str, Any] = {
        "n_sessions": args.n_sessions,
        "seed": args.seed,
        "t_max": args.t_max,
        "sequential": args.sequential,
        "dynamics": dynamics,
        "policies": {},
    }
    for name, policy in policies.items():
        rng = random.Random(args.seed)  # same seed per policy → paired comparison
        stats = run_policy(
            policy, archetypes, mixture_prior,
            n_sessions=args.n_sessions, rng=rng, t_max=args.t_max,
            dynamics=dynamics, sequential=args.sequential,
        )
        results["policies"][name] = stats

    noop = results["policies"]["no_op"]["mean_reward_per_session"]
    uni = results["policies"]["uniform_random"]["mean_reward_per_session"]
    denom = uni - noop

    print(f"\nSimulator-oracle evaluation  ({args.n_sessions:,} sessions/policy, seed={args.seed})")
    print(f"{'POLICY':<24}{'MEAN_REWARD':>13}{'STD':>9}{'PURCH~':>9}{'LIFT_vs_noop':>14}")
    print("-" * 69)
    for name, st in results["policies"].items():
        mean = st["mean_reward_per_session"]
        lift = (mean - noop) / denom if abs(denom) > 1e-9 else float("nan")
        st["lift_vs_noop_normalized"] = lift
        print(f"{name:<24}{mean:>13.4f}{st['std_reward_per_session']:>9.3f}"
              f"{st['purchase_rate']:>9.3f}{lift:>14.3f}")
    print("\n(lift: 0=no_op floor, 1=uniform-random baseline; >1 beats random)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sim_eval_all_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWritten: {out / 'sim_eval_all_results.json'}")


if __name__ == "__main__":
    main()
