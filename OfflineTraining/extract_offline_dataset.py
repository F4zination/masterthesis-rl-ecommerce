#!/usr/bin/env python3
"""Build an offline RL dataset from DemoSite_Bandis SQLite logs.

This script reconstructs per-decision rewards from decision logs + events and
emits a transition dataset with:
    (s_t, a_t, r_t, p_t, s_{t+1}, done)

It mirrors the reward logic in DemoSite_Bandis/app/services/decision.py.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from shared_schema.constants import ACTION_COST
from shared_schema.features import _decision_point_from_page
from shared_schema.features import _event_reward as event_reward


OPPORTUNITY_STATE_KEYS = (
    "opportunity_id",
    "opportunity_type",
    "opportunity_index",
    "elapsed_ms",
    "time_since_last_decision_ms",
    "next_earliest_ms",
)


@dataclass
class Decision:
    id: int
    session_id: str
    decision_point: str
    action: str
    propensity: float
    context_json: dict[str, Any]
    timestamp: datetime


@dataclass
class Event:
    id: int
    session_id: str
    event_type: str
    page: str
    metadata_json: dict[str, Any]
    timestamp: datetime


def parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value).replace(" ", "T")
    return datetime.fromisoformat(text)


def parse_dt_arg(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace(" ", "T"))


def parse_json_obj(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def load_decisions(
    conn: sqlite3.Connection,
    min_timestamp: datetime | None = None,
    max_timestamp: datetime | None = None,
) -> list[Decision]:
    where = []
    params: list[str] = []
    if min_timestamp is not None:
        where.append("timestamp >= ?")
        params.append(min_timestamp.isoformat())
    if max_timestamp is not None:
        where.append("timestamp < ?")
        params.append(max_timestamp.isoformat())

    sql = """
        SELECT id, session_id, decision_point, action, propensity, context_json, timestamp
        FROM decision_logs
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY session_id, timestamp, id"

    rows = conn.execute(sql, params).fetchall()

    decisions: list[Decision] = []
    for row in rows:
        decisions.append(
            Decision(
                id=int(row[0]),
                session_id=str(row[1]),
                decision_point=str(row[2]),
                action=str(row[3]),
                propensity=float(row[4]),
                context_json=parse_json_obj(row[5]),
                timestamp=parse_dt(row[6]),
            )
        )
    return decisions


def load_events(
    conn: sqlite3.Connection,
    min_timestamp: datetime | None = None,
    max_timestamp: datetime | None = None,
) -> list[Event]:
    where = []
    params: list[str] = []
    if min_timestamp is not None:
        where.append("timestamp >= ?")
        params.append(min_timestamp.isoformat())
    if max_timestamp is not None:
        where.append("timestamp < ?")
        params.append(max_timestamp.isoformat())

    sql = """
        SELECT id, session_id, event_type, page, metadata_json, timestamp
        FROM events
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY session_id, timestamp, id"

    rows = conn.execute(sql, params).fetchall()

    events: list[Event] = []
    for row in rows:
        events.append(
            Event(
                id=int(row[0]),
                session_id=str(row[1]),
                event_type=str(row[2]),
                page=str(row[3] or ""),
                metadata_json=parse_json_obj(row[4]),
                timestamp=parse_dt(row[5]),
            )
        )
    return events


def choose_state(
    context_json: dict[str, Any],
    decision_point: str,
    include_timing: bool = False,
) -> dict[str, Any]:
    normalized = context_json.get("context_normalized")
    if isinstance(normalized, dict) and normalized:
        state = dict(normalized)
    else:
        raw = context_json.get("context_raw")
        state = dict(raw) if isinstance(raw, dict) else {}
        state["decision_point"] = decision_point

    if include_timing:
        raw = context_json.get("context_raw")
        if isinstance(raw, dict):
            for key in OPPORTUNITY_STATE_KEYS:
                if key in raw:
                    state[key] = raw.get(key)

    state.setdefault("decision_point", decision_point)
    return state


def build_dataset(
    decisions: list[Decision],
    events: list[Event],
    attribution_window_seconds: int,
    include_timing: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decisions_by_session: dict[str, list[Decision]] = defaultdict(list)
    for d in decisions:
        decisions_by_session[d.session_id].append(d)

    event_used = set()
    unknown_event_types = Counter()
    reward_component_counter = Counter()
    transitions: list[dict[str, Any]] = []

    # Primary attribution by explicit decision_id in event metadata.
    decision_reward = defaultdict(float)
    for e in events:
        explicit_id = e.metadata_json.get("decision_id")
        if explicit_id is None:
            continue
        try:
            did = int(explicit_id)
        except (ValueError, TypeError):
            continue
        r = event_reward(e.event_type, e.metadata_json)
        if r != 0.0:
            decision_reward[did] += r
            reward_component_counter[e.event_type] += 1
            event_used.add(e.id)
        else:
            unknown_event_types[e.event_type] += 1

    # Secondary attribution mirrors the online path in
    # app/services/decision.py::apply_reward_from_event exactly: the *latest*
    # decision in the same session within the attribution window that matches
    # the event's explicit (metadata) or page-inferred decision point, and its
    # action when the metadata names one.  Keeping both paths identical
    # guarantees an event produces the same transition reward online (bandit
    # stat update) and offline (dataset reconstruction).
    for e in events:
        if e.id in event_used:
            continue
        r = event_reward(e.event_type, e.metadata_json)
        if r == 0.0:
            unknown_event_types[e.event_type] += 1
            continue

        explicit_point = e.metadata_json.get("decision_point")
        explicit_action = e.metadata_json.get("action")
        point = str(explicit_point) if explicit_point else _decision_point_from_page(e.page)

        chosen: Decision | None = None
        for d in reversed(decisions_by_session.get(e.session_id, [])):
            if d.timestamp > e.timestamp:
                continue
            if (e.timestamp - d.timestamp).total_seconds() > attribution_window_seconds:
                break  # decisions are sorted; everything earlier is older still
            if point and d.decision_point != point:
                continue
            if explicit_action and d.action != str(explicit_action):
                continue
            chosen = d
            break

        if chosen is not None:
            decision_reward[chosen.id] += r
            reward_component_counter[e.event_type] += 1
            event_used.add(e.id)

    for session_id, ds in decisions_by_session.items():
        for t, d in enumerate(ds):
            state = choose_state(
                d.context_json,
                d.decision_point,
                include_timing=include_timing,
            )
            eligible_actions = d.context_json.get("eligible_actions")
            if not isinstance(eligible_actions, list):
                eligible_actions = []

            base_cost = float(ACTION_COST.get(d.action, 0.0))
            reward = decision_reward[d.id] - base_cost

            next_state = None
            done = True
            if t + 1 < len(ds):
                done = False
                next_decision = ds[t + 1]
                next_state = choose_state(
                    next_decision.context_json,
                    next_decision.decision_point,
                    include_timing=include_timing,
                )

            transitions.append(
                {
                    "trajectory_id": session_id,
                    "t": t,
                    "decision_id": d.id,
                    "timestamp": d.timestamp.isoformat(),
                    "session_id": d.session_id,
                    "decision_point": d.decision_point,
                    "state": state,
                    "action": d.action,
                    "propensity": d.propensity,
                    "eligible_actions": eligible_actions,
                    "reward": reward,
                    "reward_without_cost": decision_reward[d.id],
                    "action_cost": base_cost,
                    "next_state": next_state,
                    "done": done,
                }
            )

    summary = {
        "n_transitions": len(transitions),
        "n_sessions": len(decisions_by_session),
        "n_events": len(events),
        "n_events_attributed": len(event_used),
        "n_events_unattributed": len(events) - len(event_used),
        "unknown_reward_event_types_seen": dict(unknown_event_types),
        "reward_event_components_used": dict(reward_component_counter),
        "reward_mean": sum(x["reward"] for x in transitions) / max(len(transitions), 1),
        "reward_min": min((x["reward"] for x in transitions), default=0.0),
        "reward_max": max((x["reward"] for x in transitions), default=0.0),
        "include_timing": include_timing,
    }

    return transitions, summary


def write_outputs(out_dir: Path, transitions: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "offline_transitions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in transitions:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    csv_path = out_dir / "offline_transitions_flat.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trajectory_id",
                "t",
                "decision_id",
                "timestamp",
                "session_id",
                "decision_point",
                "state_json",
                "action",
                "propensity",
                "eligible_actions_json",
                "reward",
                "reward_without_cost",
                "action_cost",
                "next_state_json",
                "done",
            ],
        )
        writer.writeheader()
        for row in transitions:
            writer.writerow(
                {
                    "trajectory_id": row["trajectory_id"],
                    "t": row["t"],
                    "decision_id": row["decision_id"],
                    "timestamp": row["timestamp"],
                    "session_id": row["session_id"],
                    "decision_point": row["decision_point"],
                    "state_json": json.dumps(row["state"], ensure_ascii=True, sort_keys=True),
                    "action": row["action"],
                    "propensity": row["propensity"],
                    "eligible_actions_json": json.dumps(row["eligible_actions"], ensure_ascii=True),
                    "reward": row["reward"],
                    "reward_without_cost": row["reward_without_cost"],
                    "action_cost": row["action_cost"],
                    "next_state_json": json.dumps(row["next_state"], ensure_ascii=True, sort_keys=True),
                    "done": row["done"],
                }
            )

    summary_path = out_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract offline transitions from DemoSiteV3 SQLite logs")
    parser.add_argument(
        "--db-path",
        default=str(Path(__file__).resolve().parents[1] / "DemoSiteV3" / "demosite_test.db"),
        help="Path to SQLite database (default: ../DemoSiteV3/demosite_test.db)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "data"),
        help="Output directory for extracted dataset",
    )
    parser.add_argument(
        "--attribution-window-seconds",
        type=int,
        default=1800,
        help="Max seconds after a decision to attribute unmatched events",
    )
    parser.add_argument(
        "--timing-mode",
        choices=["legacy", "opportunity"],
        default="legacy",
        help="legacy keeps old state schema, opportunity adds timing/opportunity fields from context_raw",
    )
    parser.add_argument(
        "--min-timestamp",
        default=None,
        help="Optional inclusive lower bound ISO timestamp for decision/event extraction",
    )
    parser.add_argument(
        "--max-timestamp",
        default=None,
        help="Optional exclusive upper bound ISO timestamp for decision/event extraction",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    min_ts = parse_dt_arg(args.min_timestamp)
    max_ts = parse_dt_arg(args.max_timestamp)

    conn = sqlite3.connect(str(db_path))
    try:
        decisions = load_decisions(conn, min_timestamp=min_ts, max_timestamp=max_ts)
        events = load_events(conn, min_timestamp=min_ts, max_timestamp=max_ts)
    finally:
        conn.close()

    transitions, summary = build_dataset(
        decisions=decisions,
        events=events,
        attribution_window_seconds=args.attribution_window_seconds,
        include_timing=args.timing_mode == "opportunity",
    )

    out_dir = Path(args.out_dir)
    summary.update(
        {
            "min_timestamp": min_ts.isoformat() if min_ts else None,
            "max_timestamp": max_ts.isoformat() if max_ts else None,
        }
    )
    write_outputs(out_dir, transitions, summary)

    print("[extract] done")
    print(json.dumps({"db_path": str(db_path), **summary}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
