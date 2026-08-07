#!/usr/bin/env python3
"""Offline policy evaluation utilities (IPS, SNIPS, DR) for DemoSite logs."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from generate_html_report import generate_html_report


def state_key(state: dict[str, Any]) -> str:
    return json.dumps(state, sort_keys=True, ensure_ascii=True)


def load_transitions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_reward_model(rows: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    agg_sum = defaultdict(float)
    agg_cnt = defaultdict(int)
    for r in rows:
        key = (state_key(r["state"]), str(r["action"]))
        agg_sum[key] += float(r["reward"])
        agg_cnt[key] += 1

    qhat = {}
    for key, total in agg_sum.items():
        qhat[key] = total / max(agg_cnt[key], 1)
    return qhat


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(var)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(round((len(s) - 1) * p))
    idx = max(0, min(idx, len(s) - 1))
    return s[idx]


def behavior_policy_prob(row: dict[str, Any], action: str) -> float:
    """Probability the *behavior* policy assigned to ``action`` in this state.

    For the logged action this is exactly the logged propensity, which makes
    the IPS weight w = p/b = 1 and the IPS estimate collapse to the empirical
    mean logged reward — the correct on-policy value of the behavior policy.
    (Returning 1.0 here, as an earlier revision did, silently up-weights
    low-propensity rows and no longer estimates any policy's value.)

    For non-logged actions the behavior probability is unknown from the log;
    returning 0.0 makes DM/DR terms for the behavior target a lower-bound
    approximation — acceptable for its role as a sanity baseline.
    """
    logged_action = str(row["action"])
    if action == logged_action:
        return float(row.get("propensity") or 0.0)
    return 0.0


def no_op_policy_prob(row: dict[str, Any], action: str) -> float:
    eligible = row.get("eligible_actions") or []
    target = "no-op" if "no-op" in eligible else str(row["action"])
    return 1.0 if action == target else 0.0


def greedy_empirical_policy_builder(rows: list[dict[str, Any]]) -> Callable[[dict[str, Any], str], float]:
    sums = defaultdict(float)
    counts = defaultdict(int)
    actions_by_state = defaultdict(set)

    for r in rows:
        sk = state_key(r["state"])
        a = str(r["action"])
        sums[(sk, a)] += float(r["reward"])
        counts[(sk, a)] += 1
        actions_by_state[sk].add(a)

    best_action = {}
    for sk, actions in actions_by_state.items():
        scored = []
        for a in actions:
            scored.append((sums[(sk, a)] / max(counts[(sk, a)], 1), a))
        scored.sort(reverse=True)
        best_action[sk] = scored[0][1]

    def _prob(row: dict[str, Any], action: str) -> float:
        sk = state_key(row["state"])
        target = best_action.get(sk, "no-op")
        return 1.0 if action == target else 0.0

    return _prob


def policy_from_file(policy_path: Path, temperature: float = 0.0) -> Callable[[dict[str, Any], str], float]:
    """Load a trained policy from JSON.

    When temperature > 0, returns softmax probabilities over Q-values instead
    of a deterministic 0/1 indicator.  This gives non-zero importance weights
    for all logged actions, reducing IPS/DR variance from zero-weight problem.
    temperature=0 (default) retains the original greedy/deterministic behaviour.
    """
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    explicit = payload.get("policy")
    default_by_point = payload.get("default_action_by_decision_point", {})

    if not isinstance(explicit, dict):
        raise ValueError("Policy file must contain a top-level 'policy' object")

    # Parse Q-table for softmax mode; keys are "state_json|||action"
    q_by_state: dict[str, dict[str, float]] = defaultdict(dict)
    for key, qval in (payload.get("q_table") or {}).items():
        parts = key.split("|||", 1)
        if len(parts) == 2:
            q_by_state[parts[0]][parts[1]] = float(qval)

    def _prob(row: dict[str, Any], action: str) -> float:
        sk = state_key(row["state"])
        point = str(row.get("decision_point") or "")

        if temperature <= 0.0 or not q_by_state.get(sk):
            # Greedy / deterministic fallback
            target = explicit.get(sk) or default_by_point.get(point) or "no-op"
            return 1.0 if action == target else 0.0

        # Softmax over Q-values for this state
        q_vals = q_by_state[sk]
        eligible = list(row.get("eligible_actions") or q_vals.keys())
        scores = [q_vals.get(a, 0.0) / temperature for a in eligible]
        max_s = max(scores)
        exps = [math.exp(s - max_s) for s in scores]
        total = sum(exps)
        action_probs = {a: e / total for a, e in zip(eligible, exps)}
        return action_probs.get(action, 0.0)

    return _prob


def evaluate_policy(
    rows: list[dict[str, Any]],
    policy_prob: Callable[[dict[str, Any], str], float],
    qhat: dict[tuple[str, str], float],
) -> dict[str, float]:
    weighted = []
    dr_terms = []
    dm_terms = []
    w_values = []

    for r in rows:
        a = str(r["action"])
        bprob = float(r["propensity"])
        if bprob <= 0:
            continue

        p = float(policy_prob(r, a))
        w = p / bprob

        rew = float(r["reward"])
        sk = state_key(r["state"])
        q_sa = qhat.get((sk, a), 0.0)

        target_v = 0.0
        eligible = r.get("eligible_actions") or [a]
        for cand in eligible:
            pi = float(policy_prob(r, str(cand)))
            target_v += pi * qhat.get((sk, str(cand)), 0.0)

        weighted.append(w * rew)
        w_values.append(w)
        dr_terms.append(target_v + w * (rew - q_sa))
        dm_terms.append(target_v)

    n = max(len(weighted), 1)
    ips = sum(weighted) / n
    snips = sum(weighted) / max(sum(w_values), 1e-12)
    dr = sum(dr_terms) / max(len(dr_terms), 1)
    dm = sum(dm_terms) / max(len(dm_terms), 1)

    return {
        "ips": ips,
        "snips": snips,
        "dr": dr,
        "dm": dm,
        "mean_weight": mean(w_values),
        "max_weight": max(w_values) if w_values else 0.0,
        "n": len(weighted),
    }


def bootstrap_ci(
    rows: list[dict[str, Any]],
    policy_prob: Callable[[dict[str, Any], str], float],
    qhat: dict[tuple[str, str], float],
    n_bootstrap: int,
    seed: int,
) -> dict[str, list[float]]:
    if not rows:
        return {"ips": [0.0, 0.0], "snips": [0.0, 0.0], "dr": [0.0, 0.0], "dm": [0.0, 0.0]}

    rng = random.Random(seed)
    ips_vals = []
    snips_vals = []
    dr_vals = []
    dm_vals = []

    for _ in range(n_bootstrap):
        sample = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        est = evaluate_policy(sample, policy_prob, qhat)
        ips_vals.append(est["ips"])
        snips_vals.append(est["snips"])
        dr_vals.append(est["dr"])
        dm_vals.append(est["dm"])

    return {
        "ips": [percentile(ips_vals, 0.025), percentile(ips_vals, 0.975)],
        "snips": [percentile(snips_vals, 0.025), percentile(snips_vals, 0.975)],
        "dr": [percentile(dr_vals, 0.025), percentile(dr_vals, 0.975)],
        "dm": [percentile(dm_vals, 0.025), percentile(dm_vals, 0.975)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline policy evaluation on extracted transitions")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parent / "data" / "offline_transitions.jsonl"),
        help="Eval transitions JSONL (scored by all policies)",
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        help="Optional separate training JSONL used to build qhat and greedy_empirical. "
             "When provided, those are fit on train data and evaluated on --dataset (eval data), "
             "removing in-sample evaluation bias.",
    )
    parser.add_argument(
        "--policy-file",
        default=None,
        help="Optional policy JSON produced by train_offline_policy.py",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Softmax temperature for trained_policy probability. "
             "0 (default) = greedy/deterministic (original behaviour). "
             ">0 spreads probability over all actions via softmax(Q/T), "
             "giving non-zero IPS weights for transitions where policy disagrees with log.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="Number of bootstrap replicates for 95 percent CI",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "outputs"),
        help="Directory for evaluation artifacts",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"[ope] Loading eval transitions from {dataset_path}")
    rows = load_transitions(dataset_path)
    print(f"[ope] Loaded {len(rows):,} eval transitions")

    # Separate training set for reward model + greedy empirical (removes in-sample bias)
    if args.train_dataset:
        train_path = Path(args.train_dataset)
        if not train_path.exists():
            raise FileNotFoundError(f"Train dataset not found: {train_path}")
        print(f"[ope] Loading train transitions from {train_path}")
        train_rows = load_transitions(train_path)
        print(f"[ope] Loaded {len(train_rows):,} train transitions")
    else:
        train_rows = rows

    print("[ope] Building reward model (Q-hat)")
    qhat = build_reward_model(train_rows)
    print(f"[ope] Reward model: {len(qhat):,} (state, action) pairs")

    policies: dict[str, Callable[[dict[str, Any], str], float]] = {
        "behavior": behavior_policy_prob,
        "no_op": no_op_policy_prob,
        "greedy_empirical": greedy_empirical_policy_builder(train_rows),
    }

    if args.policy_file:
        print(f"[ope] Loading trained policy from {args.policy_file} (temperature={args.temperature})")
        policies["trained_policy"] = policy_from_file(Path(args.policy_file), temperature=args.temperature)

    print(f"[ope] Evaluating {len(policies)} policies with {args.bootstrap} bootstrap replicates each")

    results: dict[str, Any] = {
        "dataset": str(dataset_path),
        "train_dataset": str(args.train_dataset) if args.train_dataset else None,
        "temperature": args.temperature,
        "n_rows": len(rows),
        "bootstrap": args.bootstrap,
        "policies": {},
    }

    for name, fn in policies.items():
        print(f"[ope]   evaluating '{name}'...")
        point_est = evaluate_policy(rows, fn, qhat)
        ci = bootstrap_ci(rows, fn, qhat, n_bootstrap=args.bootstrap, seed=args.seed)
        results["policies"][name] = {
            "point_estimate": point_est,
            "ci95": ci,
        }
        print(f"[ope]   '{name}' done  — DR={point_est['dr']:.4f}  DM={point_est['dm']:.4f}  IPS={point_est['ips']:.4f}  SNIPS={point_est['snips']:.4f}")

    # Normalized lift: (trained_DR - noop_DR) / (behavior_DR - noop_DR)
    # >1.0 means trained policy outperforms the behavior policy relative to the no-op floor.
    policy_results = results["policies"]
    if all(k in policy_results for k in ("trained_policy", "no_op", "behavior")):
        trained_dr = policy_results["trained_policy"]["point_estimate"]["dr"]
        noop_dr = policy_results["no_op"]["point_estimate"]["dr"]
        behavior_dr = policy_results["behavior"]["point_estimate"]["dr"]
        denom = behavior_dr - noop_dr
        lift = (trained_dr - noop_dr) / denom if abs(denom) > 1e-9 else float("nan")
        results["lift_vs_noop_normalized"] = lift
        print(f"[ope] Normalized lift (DR):  {lift:.4f}  (>1.0 = beats behavior policy relative to no-op floor)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_out = out_dir / "ope_results.json"
    json_out.write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[ope] Results written to {json_out}")

    # Compact markdown report for thesis notes.
    lines = ["# Offline Policy Evaluation", "", f"Dataset: {dataset_path}", f"Rows: {len(rows)}", ""]
    if args.train_dataset:
        lines.insert(3, f"Train dataset: {args.train_dataset}")
    for name, payload in results["policies"].items():
        pe = payload["point_estimate"]
        ci = payload["ci95"]
        lines.append(f"## Policy: {name}")
        lines.append(f"- IPS: {pe['ips']:.4f}  (95 percent CI {ci['ips'][0]:.4f}, {ci['ips'][1]:.4f})")
        lines.append(f"- SNIPS: {pe['snips']:.4f}  (95 percent CI {ci['snips'][0]:.4f}, {ci['snips'][1]:.4f})")
        lines.append(f"- DR: {pe['dr']:.4f}  (95 percent CI {ci['dr'][0]:.4f}, {ci['dr'][1]:.4f})")
        lines.append(f"- DM: {pe['dm']:.4f}  (95 percent CI {ci['dm'][0]:.4f}, {ci['dm'][1]:.4f})")
        lines.append(f"- Mean importance weight: {pe['mean_weight']:.4f}")
        lines.append(f"- Max importance weight: {pe['max_weight']:.4f}")
        lines.append("")
    if "lift_vs_noop_normalized" in results:
        lines.append(f"## Normalized Lift (DR): {results['lift_vs_noop_normalized']:.4f}")

    md_out = out_dir / "ope_report.md"
    md_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ope] Markdown report written to {md_out}")

    # Auto-generate HTML report
    policy_file = args.policy_file if args.policy_file else None
    dataset_summary_path = dataset_path.parent / "dataset_summary.json"
    try:
        report_path = generate_html_report(
            output_dir=str(out_dir),
            policy_path=str(policy_file) if policy_file else None,
            dataset_path=str(dataset_summary_path) if dataset_summary_path.exists() else None,
            ope_path=str(json_out),
        )
        print(f"[ope] Report: {report_path}")
    except Exception as e:
        print(f"[ope] Warning: Could not generate HTML report: {e}")

    print("[ope] done")
    print(json.dumps(results, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
