#!/usr/bin/env python3
"""Train a simple offline tabular Q policy from extracted transitions.

This is a practical first Phase-2 baseline for small datasets.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from generate_html_report import generate_html_report


def state_key(state: dict[str, Any]) -> str:
    return json.dumps(state, sort_keys=True, ensure_ascii=True)


def load_transitions(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_action_support(rows: list[dict[str, Any]]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    actions_by_state: dict[str, set[str]] = defaultdict(set)
    actions_by_point: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        sk = state_key(r["state"])
        a = str(r["action"])
        point = str(r.get("decision_point") or "")
        actions_by_state[sk].add(a)
        actions_by_point[point].add(a)

        eligible = r.get("eligible_actions") or []
        for e in eligible:
            actions_by_state[sk].add(str(e))
            actions_by_point[point].add(str(e))
    return actions_by_state, actions_by_point


def fitted_q_iteration(
    rows: list[dict[str, Any]],
    gamma: float,
    iters: int,
    conservative_penalty: float,
) -> dict[tuple[str, str], float]:
    q = defaultdict(float)
    sa_count = defaultdict(int)

    actions_by_state, _ = build_action_support(rows)
    for r in rows:
        sa_count[(state_key(r["state"]), str(r["action"]))] += 1

    for _ in range(iters):
        targets = defaultdict(list)

        for r in rows:
            s = state_key(r["state"])
            a = str(r["action"])
            reward = float(r["reward"])

            if bool(r.get("done", True)) or not isinstance(r.get("next_state"), dict):
                target = reward
            else:
                ns = state_key(r["next_state"])
                next_actions = actions_by_state.get(ns, set())
                if next_actions:
                    max_q_next = max(q[(ns, na)] for na in next_actions)
                else:
                    max_q_next = 0.0
                target = reward + gamma * max_q_next

            n_sa = sa_count[(s, a)]
            penalty = conservative_penalty / max(n_sa ** 0.5, 1.0)
            targets[(s, a)].append(target - penalty)

        for key, vals in targets.items():
            q[key] = sum(vals) / max(len(vals), 1)

    return dict(q)


def greedy_policy(
    q: dict[tuple[str, str], float],
    rows: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    actions_by_state, actions_by_point = build_action_support(rows)

    point_defaults = {}
    for point, actions in actions_by_point.items():
        if "no-op" in actions:
            point_defaults[point] = "no-op"
        else:
            point_defaults[point] = sorted(actions)[0] if actions else "no-op"

    policy = {}
    for r in rows:
        sk = state_key(r["state"])
        point = str(r.get("decision_point") or "")
        actions = actions_by_state.get(sk, set())
        if not actions:
            policy[sk] = point_defaults.get(point, "no-op")
            continue

        best_a = None
        best_q = float("-inf")
        # Sorted iteration makes the argmax tie-break independent of set/hash
        # order, so a run is bit-reproducible across processes (PYTHONHASHSEED).
        for a in sorted(actions):
            qa = q.get((sk, a), 0.0)
            if qa > best_q:
                best_q = qa
                best_a = a
        policy[sk] = best_a if best_a is not None else point_defaults.get(point, "no-op")

    return policy, point_defaults


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tabular offline Q policy")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parent / "data" / "offline_transitions.jsonl"),
        help="Path to extracted transitions JSONL",
    )
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--conservative-penalty",
        type=float,
        default=0.15,
        help="Penalty for low-support state-action pairs",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "outputs"),
        help="Directory for trained policy artifacts",
    )
    parser.add_argument(
        "--timing-mode",
        choices=["legacy", "opportunity"],
        default="legacy",
        help="Metadata label for the trained artifact to indicate state schema mode",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rows = load_transitions(dataset_path)
    q = fitted_q_iteration(
        rows=rows,
        gamma=args.gamma,
        iters=args.iters,
        conservative_penalty=args.conservative_penalty,
    )
    policy, defaults = greedy_policy(q, rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    q_table_serialized = {
        f"{k[0]}|||{k[1]}": v
        for k, v in sorted(q.items(), key=lambda kv: kv[1], reverse=True)
    }

    payload = {
        "algorithm": "tabular_fqi_conservative",
        "timing_mode": args.timing_mode,
        "gamma": args.gamma,
        "iters": args.iters,
        "conservative_penalty": args.conservative_penalty,
        "n_rows": len(rows),
        "n_states": len({state_key(r["state"]) for r in rows}),
        "policy": policy,
        "default_action_by_decision_point": defaults,
        "q_table": q_table_serialized,
    }

    out_path = out_dir / "trained_policy.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    # Auto-generate HTML report
    dataset_summary_path = dataset_path.parent / "dataset_summary.json"
    try:
        report_path = generate_html_report(
            output_dir=str(out_dir),
            policy_path=str(out_path),
            dataset_path=str(dataset_summary_path) if dataset_summary_path.exists() else None,
        )
        print(f"[train] Report: {report_path}")
    except Exception as e:
        print(f"[train] Warning: Could not generate HTML report: {e}")

    print("[train] done")
    print(json.dumps({"out_path": str(out_path), "n_policy_states": len(policy)}, indent=2))


if __name__ == "__main__":
    main()
