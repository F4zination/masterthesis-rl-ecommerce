#!/usr/bin/env python3
"""Analyze the frozen-policy Clickworker study at participant/session level.

The script reads the V2 and V3 SQLite databases directly and produces one
analysis row per randomized participant.  When the dispatcher ledger is
provided, every assignment is retained in the intention-to-treat (ITT) data,
including assignments with no observed shop session.  If a participant opened
more than one shop session, the first observed session is the pre-specified ITT
session and later sessions are reported as duplicates.

The policy estimand is the equal-persona average of the within-persona V3-V2
difference.  Inference uses a stratified randomization test and a bootstrap
that resamples participants within the ten policy x persona cells.  No
action-level OPE is attempted: the deployed policies are deterministic and do
not provide the overlap required for IPS/SNIPS/DR comparisons.

Example::

    python Experiments/analyze_clickworker_study.py \
        --db v2=DemoSiteV2/data/demosite.db \
        --db v3=DemoSiteV3/data/demosite.db \
        --dispatcher-db StudyDispatcher/data/dispatcher.db \
        --out-dir Experiments/clickworker_analysis
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "SharedSchema"))

from shared_schema.constants import ACTION_COST  # noqa: E402
from shared_schema.features import (  # noqa: E402
    _device_type,
    _event_reward as shared_event_reward,
    _traffic_source,
)


ALL_PERSONAS = (
    "explorer",
    "fastbuyer",
    "detailedcomparator",
    "discounthunter",
    "windowshopper",
)
# The recruited subset, narrowed by --personas at startup.  A design may assign
# fewer archetypes than the locked simulator reference evaluates (Design
# Revision 1 recruits three of five), and every estimand that averages over
# policy x persona cells must then average over the recruited cells only.
# Cells outside this set are absent by design, not missing data, so leaving the
# full set in place makes those estimands unresolvable rather than restricted.
PERSONAS = ALL_PERSONAS
CONDITIONS = ("v2", "v3")
DEVICE_STRATA = ("unknown", "mobile", "tablet", "desktop")
TRAFFIC_STRATA = ("direct", "search", "social", "referral")
BASELINE_POLICY = {"v2": "v2_bandit", "v3": "v3_ppo"}
BASELINE_ARCHETYPE = {
    "explorer": "Explorer",
    "fastbuyer": "FastBuyer",
    "detailedcomparator": "DetailedComparator",
    "discounthunter": "DiscountHunter",
    "windowshopper": "WindowShopper",
}
FIDELITY_OUTCOMES = {
    "conversion": ("conversion_rate", "conversion_sd", 0.05),
    "session_length_steps": ("step_count_mean", "step_count_sd", 0.50),
    "funnel_depth": ("funnel_depth_mean", "funnel_depth_sd", 0.50),
}
MIN_DURATION_S = 90.0
MIN_PAGE_VIEWS = 3

# ---------------------------------------------------------------------------
# Transition-ordering endpoint (August 2026 redesign).
#
# The primary claim of the redesigned study is directional: the simulator's
# predicted *ordering* of the conditional funnel-advance proportions across
# the recruited personas holds in humans. The observable on both sides is the
# session-level funnel progression P(ever reach stage k+1 | ever reached
# stage k), derived identically from max funnel depth, so the comparison is
# definitionally aligned with the simulator reference's funnel_progression
# emission. It is not the per-step Markov parameter in archetypes.yaml.
FUNNEL_PROGRESSION_PAIRS = (
    ("landing", "pdp"),
    ("pdp", "cart"),
    ("cart", "checkout"),
    ("checkout", "purchase"),
)

# Confirmatory contrasts, each written (stage_pair, higher, lower) with the
# direction fixed a priori by the locked simulator reference. The verdict is
# intersection-union: all three must hold at one-sided alpha = 0.05, so no
# multiplicity adjustment is required. Every other persona pair is reported
# descriptively because the simulator's own gap there is too narrow to be
# detectable at a convenience-sample N; a null on those pairs is
# uninformative rather than evidence against the simulator.
#
# Selected by applying the preregistered rule -- confirmatory pairs are those
# the reference separates widely enough for a null to be interpretable -- to
# the locked reference clickworker_release_20260803. Note that PDP->cart
# fastbuyer vs detailedcomparator is *descriptive* despite a wide per-step
# configured gap (0.55 vs 0.18): at the session level DetailedComparator
# revisits product pages repeatedly and so gets many chances to reach the
# cart, lifting its "ever advanced" proportion to 0.417 against FastBuyer's
# 0.596. The realized gap is 0.178, which needs ~360 participants.
CONFIRMATORY_ORDERING_CONTRASTS = (
    ("landing_to_pdp", "fastbuyer", "windowshopper"),
    ("landing_to_pdp", "detailedcomparator", "windowshopper"),
    ("pdp_to_cart", "fastbuyer", "windowshopper"),
)
ORDERING_ALPHA_ONE_SIDED = 0.05
# Below this many participants in a conditioning set, the contrast is reported
# but not tested: a proportion estimated from a handful of sessions carries no
# directional information worth a p-value.
MIN_REACHED_FOR_ORDERING_TEST = 10


def _json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value in (None, ""):
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text.replace(" ", "T"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except ValueError:
        return None


def parse_cutoff(value: str) -> datetime:
    """Parse a recruitment-window bound into a naive-UTC exclusive upper bound.

    A bare ``YYYY-MM-DD`` names a day that is kept in full, so the bound is the
    following midnight; an explicit datetime is used as given.  Session and
    assignment timestamps are compared in UTC.
    """
    text = value.strip().replace("Z", "+00:00")
    date_only = ":" not in text
    try:
        parsed = datetime.fromisoformat(text.replace(" ", "T"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected an ISO date or datetime, got {value!r}"
        ) from error
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed + timedelta(days=1) if date_only else parsed


def apply_cutoff(
    sessions: Sequence[RawSession],
    assignments: Sequence[Assignment] | None,
    cutoff: datetime,
) -> tuple[list[RawSession], list[Assignment] | None, int, int]:
    """Truncate sessions and assignments at an exclusive naive-UTC bound.

    Sessions carrying no parsable timestamp cannot be placed in time and are
    left to the attribution audit rather than silently dropped.
    """
    cutoff_epoch = cutoff.replace(tzinfo=timezone.utc).timestamp()
    kept_sessions = [
        session for session in sessions
        if not session.timestamps or session.first_seen < cutoff
    ]
    if assignments is None:
        return kept_sessions, None, len(sessions) - len(kept_sessions), 0
    kept_assignments = [
        assignment for assignment in assignments
        if assignment.assigned_at is None or assignment.assigned_at < cutoff_epoch
    ]
    return (
        kept_sessions,
        kept_assignments,
        len(sessions) - len(kept_sessions),
        len(assignments) - len(kept_assignments),
    )


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


@dataclass
class RawSession:
    condition: str
    session_id: str
    worker_id: str = ""
    persona: str = ""
    device_type: str = ""
    traffic_source: str = ""
    timestamps: list[datetime] = field(default_factory=list)
    event_types: list[str] = field(default_factory=list)
    event_pages: list[str] = field(default_factory=list)
    event_reward: float = 0.0
    decision_actions: list[str] = field(default_factory=list)
    policy_sources: list[str] = field(default_factory=list)
    order_totals: list[float] = field(default_factory=list)
    attention_seen: bool = False
    attention_passed: bool = False

    @property
    def first_seen(self) -> datetime:
        return min(self.timestamps) if self.timestamps else datetime.max


@dataclass
class Assignment:
    worker_id: str
    condition: str
    persona: str
    assigned_at: float | None = None


@dataclass
class ParticipantRecord:
    condition: str
    worker_id: str
    persona: str
    device_type: str
    traffic_source: str
    session_id: str
    assigned: bool
    observed_session: bool
    recorded_landing: bool
    duplicate_sessions: int
    completed_attention: bool
    attention_passed: bool
    quality_ok: bool
    quality_reasons: str
    page_views: int
    duration_s: float
    event_count: int
    decision_count: int
    session_length_steps: int
    funnel_depth: int
    conversion: int
    order_total: float
    event_reward: float
    action_cost: float
    session_reward: float
    widget_clicks: int
    widget_dismissals: int
    decision_errors: int
    fallback_decisions: int
    fallback_rate: float
    attribution_valid: bool


def load_shop_sessions(condition: str, db_path: Path) -> list[RawSession]:
    """Load all observable sessions from one shop database."""
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported condition {condition!r}; expected v2 or v3")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    sessions: dict[str, RawSession] = {}

    def get(sid: str) -> RawSession:
        if sid not in sessions:
            sessions[sid] = RawSession(condition=condition, session_id=sid)
        return sessions[sid]

    if _table_exists(conn, "events"):
        for row in conn.execute(
            "SELECT session_id, event_type, page, timestamp, metadata_json, "
            "worker_id, persona FROM events ORDER BY timestamp, id"
        ):
            sid = str(row["session_id"] or "")
            if not sid:
                continue
            session = get(sid)
            session.worker_id = session.worker_id or str(row["worker_id"] or "")
            session.persona = session.persona or str(row["persona"] or "").lower()
            ts = _timestamp(row["timestamp"])
            if ts:
                session.timestamps.append(ts)
            event_type = str(row["event_type"] or "")
            page = str(row["page"] or "")
            metadata = _json(row["metadata_json"])
            if not session.device_type and "screen_width" in metadata:
                session.device_type = _device_type(metadata)
            if not session.traffic_source and "referrer" in metadata:
                session.traffic_source = _traffic_source(metadata)
            session.event_types.append(event_type)
            session.event_pages.append(page)
            session.event_reward += shared_event_reward(event_type, metadata)
            if event_type == "attention_check":
                session.attention_seen = True
                session.attention_passed = bool(metadata.get("passed"))

    if _table_exists(conn, "decision_logs"):
        for row in conn.execute(
            "SELECT session_id, action, timestamp, context_json "
            "FROM decision_logs ORDER BY timestamp, id"
        ):
            sid = str(row["session_id"] or "")
            if not sid:
                continue
            session = get(sid)
            context = _json(row["context_json"])
            normalized = context.get("context_normalized") or {}
            if not session.device_type and isinstance(normalized, dict):
                session.device_type = str(normalized.get("device_type") or "")
            if not session.traffic_source and isinstance(normalized, dict):
                session.traffic_source = str(normalized.get("traffic_source") or "")
            session.worker_id = session.worker_id or str(context.get("worker_id") or "")
            session.persona = session.persona or str(context.get("persona") or "").lower()
            ts = _timestamp(row["timestamp"])
            if ts:
                session.timestamps.append(ts)
            session.decision_actions.append(str(row["action"] or "no-op"))
            source = str(context.get("policy_source") or context.get("model") or "")
            session.policy_sources.append(source)

    if _table_exists(conn, "orders"):
        for row in conn.execute(
            "SELECT session_id, total, created_at, worker_id, persona "
            "FROM orders ORDER BY created_at, id"
        ):
            sid = str(row["session_id"] or "")
            if not sid:
                continue
            session = get(sid)
            session.worker_id = session.worker_id or str(row["worker_id"] or "")
            session.persona = session.persona or str(row["persona"] or "").lower()
            ts = _timestamp(row["created_at"])
            if ts:
                session.timestamps.append(ts)
            session.order_totals.append(float(row["total"] or 0.0))

    conn.close()
    return list(sessions.values())


def load_assignments(dispatcher_db: Path) -> list[Assignment]:
    """Load the persistent balanced-randomization ledger."""
    conn = sqlite3.connect(str(dispatcher_db))
    conn.row_factory = sqlite3.Row
    if not _table_exists(conn, "assignments"):
        conn.close()
        raise ValueError(f"assignments table missing from {dispatcher_db}")
    rows = conn.execute(
        "SELECT wid, condition, persona, created_at FROM assignments ORDER BY created_at"
    ).fetchall()
    conn.close()
    return [
        Assignment(
            worker_id=str(row["wid"] or ""),
            condition=str(row["condition"] or "").lower(),
            persona=str(row["persona"] or "").lower(),
            assigned_at=float(row["created_at"]) if row["created_at"] is not None else None,
        )
        for row in rows
    ]


def _quality(session: RawSession | None) -> tuple[bool, list[str]]:
    if session is None:
        return False, ["no observed session"]
    reasons: list[str] = []
    page_views = session.event_types.count("page_view")
    event_times = session.timestamps
    duration = (
        (max(event_times) - min(event_times)).total_seconds()
        if len(event_times) > 1
        else 0.0
    )
    engagement = any(
        event == "widget_dismiss" or event.startswith("dwell") or event.startswith("scroll")
        for event in session.event_types
    )
    if not session.attention_passed:
        reasons.append("attention not passed")
    if duration < MIN_DURATION_S:
        reasons.append(f"duration<{MIN_DURATION_S:.0f}s")
    if page_views < MIN_PAGE_VIEWS:
        reasons.append(f"page_views<{MIN_PAGE_VIEWS}")
    if not engagement:
        reasons.append("no dwell/scroll/dismiss")
    return not reasons, reasons


def _funnel_depth(session: RawSession | None) -> int:
    if session is None:
        return 0
    if session.order_totals or "purchase" in session.event_types:
        return 5
    if "checkout_submit" in session.event_types or any("checkout" in p for p in session.event_pages):
        return 4
    if "add_to_cart" in session.event_types or any("cart" in p for p in session.event_pages):
        return 3
    if any("product" in p or "pdp" in p for p in session.event_pages):
        return 2
    return 1 if session.event_types or session.decision_actions else 0


def _record(
    assignment: Assignment,
    session: RawSession | None,
    duplicate_sessions: int,
    *,
    assigned: bool,
) -> ParticipantRecord:
    quality_ok, quality_reasons = _quality(session)
    if session is None:
        return ParticipantRecord(
            condition=assignment.condition,
            worker_id=assignment.worker_id,
            persona=assignment.persona,
            device_type="unknown",
            traffic_source="direct",
            session_id="",
            assigned=assigned,
            observed_session=False,
            recorded_landing=False,
            duplicate_sessions=duplicate_sessions,
            completed_attention=False,
            attention_passed=False,
            quality_ok=False,
            quality_reasons="; ".join(quality_reasons),
            page_views=0,
            duration_s=float("nan"),
            event_count=0,
            decision_count=0,
            session_length_steps=0,
            funnel_depth=0,
            conversion=0,
            order_total=0.0,
            event_reward=0.0,
            action_cost=0.0,
            session_reward=0.0,
            widget_clicks=0,
            widget_dismissals=0,
            decision_errors=0,
            fallback_decisions=0,
            fallback_rate=0.0,
            attribution_valid=bool(assignment.worker_id and assignment.persona in PERSONAS),
        )

    duration = (
        (max(session.timestamps) - min(session.timestamps)).total_seconds()
        if len(session.timestamps) > 1
        else 0.0
    )
    action_cost = sum(ACTION_COST.get(action, 0.0) for action in session.decision_actions)
    # V3's target source is PPO. V2's contextual bandit is its intended source.
    if assignment.condition == "v3":
        fallback = sum(source != "ppo_policy" for source in session.policy_sources)
    else:
        fallback = sum(
            bool(source) and "epsilon_greedy" not in source and source != "contextual_bandit"
            for source in session.policy_sources
        )
    decision_count = len(session.decision_actions)
    recorded_landing = any(
        event_type == "page_view" and page == "/"
        for event_type, page in zip(session.event_types, session.event_pages)
    )
    return ParticipantRecord(
        condition=assignment.condition,
        worker_id=assignment.worker_id,
        persona=assignment.persona,
        device_type=(
            session.device_type if session.device_type in DEVICE_STRATA else "unknown"
        ),
        traffic_source=(
            session.traffic_source if session.traffic_source in TRAFFIC_STRATA else "direct"
        ),
        session_id=session.session_id,
        assigned=assigned,
        observed_session=True,
        recorded_landing=recorded_landing,
        duplicate_sessions=duplicate_sessions,
        completed_attention=session.attention_seen,
        attention_passed=session.attention_passed,
        quality_ok=quality_ok,
        quality_reasons="; ".join(quality_reasons),
        page_views=session.event_types.count("page_view"),
        duration_s=max(duration, 0.0),
        event_count=len(session.event_types),
        decision_count=decision_count,
        session_length_steps=decision_count,
        funnel_depth=_funnel_depth(session),
        conversion=int(bool(session.order_totals or "purchase" in session.event_types)),
        order_total=sum(session.order_totals),
        event_reward=session.event_reward,
        action_cost=action_cost,
        session_reward=session.event_reward - action_cost,
        widget_clicks=session.event_types.count("widget_click"),
        widget_dismissals=session.event_types.count("widget_dismiss"),
        decision_errors=session.event_types.count("opportunity_error"),
        fallback_decisions=fallback,
        fallback_rate=fallback / decision_count if decision_count else 0.0,
        attribution_valid=(
            assignment.condition in CONDITIONS
            and assignment.persona in PERSONAS
            and bool(assignment.worker_id)
        ),
    )


def build_participant_records(
    sessions: Sequence[RawSession],
    assignments: Sequence[Assignment] | None = None,
) -> tuple[list[ParticipantRecord], list[dict[str, Any]]]:
    """Collapse observed/resumed sessions to one pre-specified row per worker."""
    by_worker: dict[str, list[RawSession]] = defaultdict(list)
    unattributed: list[RawSession] = []
    for session in sessions:
        if session.worker_id:
            by_worker[session.worker_id].append(session)
        else:
            unattributed.append(session)
    for values in by_worker.values():
        values.sort(key=lambda item: (item.first_seen, item.session_id))

    records: list[ParticipantRecord] = []
    audit: list[dict[str, Any]] = []
    used_workers: set[str] = set()

    if assignments is not None:
        for assignment in assignments:
            observed = by_worker.get(assignment.worker_id, [])
            chosen = observed[0] if observed else None
            records.append(
                _record(assignment, chosen, max(0, len(observed) - 1), assigned=True)
            )
            used_workers.add(assignment.worker_id)
            if chosen and chosen.condition != assignment.condition:
                audit.append({
                    "type": "condition_mismatch",
                    "worker_id": assignment.worker_id,
                    "assigned_condition": assignment.condition,
                    "observed_condition": chosen.condition,
                })

    for worker_id, observed in sorted(by_worker.items()):
        if worker_id in used_workers:
            continue
        chosen = observed[0]
        inferred = Assignment(
            worker_id=worker_id,
            condition=chosen.condition,
            persona=chosen.persona,
        )
        records.append(_record(inferred, chosen, max(0, len(observed) - 1), assigned=False))
        audit.append({"type": "observed_without_dispatcher_assignment", "worker_id": worker_id})

    for session in unattributed:
        audit.append({
            "type": "unattributed_session",
            "condition": session.condition,
            "session_id": session.session_id,
        })

    return records, audit


def _valid_records(records: Iterable[ParticipantRecord]) -> list[ParticipantRecord]:
    return [r for r in records if r.attribution_valid]


def adjusted_difference(records: Sequence[ParticipantRecord], outcome: str) -> float:
    """Return the equal-persona average within-persona V3-V2 difference."""
    effects: list[float] = []
    for persona in PERSONAS:
        v2 = [float(getattr(r, outcome)) for r in records if r.persona == persona and r.condition == "v2"]
        v3 = [float(getattr(r, outcome)) for r in records if r.persona == persona and r.condition == "v3"]
        if not v2 or not v3:
            continue
        effects.append(mean(v3) - mean(v2))
    return mean(effects) if effects else float("nan")


def robust_difference_ci(
    records: Sequence[ParticipantRecord], outcome: str, z: float = 1.959963984540054
) -> tuple[float, float]:
    """HC-style normal CI for the equal-persona saturated-cell contrast.

    With a saturated policy-by-persona model, the target is a linear contrast
    of the ten independent cell means.  Summing each cell's sample-mean
    variance therefore gives the heteroskedasticity-robust variance estimate.
    """
    estimate = adjusted_difference(records, outcome)
    if math.isnan(estimate):
        return float("nan"), float("nan")
    variance = 0.0
    complete = 0
    for persona in PERSONAS:
        for condition in CONDITIONS:
            values = [
                float(getattr(record, outcome))
                for record in records
                if record.persona == persona and record.condition == condition
            ]
            if not values:
                break
            if len(values) > 1:
                cell_mean = mean(values)
                sample_variance = sum((value - cell_mean) ** 2 for value in values) / (
                    len(values) - 1
                )
                variance += sample_variance / len(values) / (len(PERSONAS) ** 2)
        else:
            complete += 1
    if complete != len(PERSONAS):
        return float("nan"), float("nan")
    standard_error = math.sqrt(max(variance, 0.0))
    return estimate - z * standard_error, estimate + z * standard_error


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy_baseline(path: Path) -> dict[str, Any]:
    """Load and validate the frozen 2 x 5 simulation reference."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cells"), list):
        raise ValueError(f"invalid baseline payload: {path}")
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    inverse_policy = {value: key for key, value in BASELINE_POLICY.items()}
    inverse_archetype = {value: key for key, value in BASELINE_ARCHETYPE.items()}
    for cell in payload["cells"]:
        if not isinstance(cell, dict):
            continue
        condition = inverse_policy.get(str(cell.get("policy")))
        persona = inverse_archetype.get(str(cell.get("archetype")))
        if condition and persona:
            key = (condition, persona)
            if key in cells:
                raise ValueError(f"duplicate baseline cell: {key}")
            cells[key] = cell
    expected = {(condition, persona) for condition in CONDITIONS for persona in PERSONAS}
    missing = sorted(expected - set(cells))
    if missing:
        raise ValueError(f"baseline is missing cells: {missing}")
    for key, cell in cells.items():
        for mean_key in ("conversion_rate", "step_count_mean", "funnel_depth_mean"):
            if mean_key not in cell:
                raise ValueError(f"baseline cell {key} is missing {mean_key}")
        if int(cell.get("n_sessions") or 0) < 2:
            raise ValueError(f"baseline cell {key} has fewer than two sessions")

    context_cells: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    raw_context_cells = payload.get("context_strata") or []
    if not isinstance(raw_context_cells, list):
        raise ValueError("baseline context_strata must be a list")
    for cell in raw_context_cells:
        if not isinstance(cell, dict):
            continue
        condition = inverse_policy.get(str(cell.get("policy")))
        persona = inverse_archetype.get(str(cell.get("archetype")))
        device = str(cell.get("device_type") or "")
        traffic = str(cell.get("traffic_source") or "")
        if condition and persona and device in DEVICE_STRATA and traffic in TRAFFIC_STRATA:
            key = (condition, persona, device, traffic)
            if key in context_cells:
                raise ValueError(f"duplicate baseline context stratum: {key}")
            if int(cell.get("n_sessions") or 0) < 2:
                raise ValueError(f"baseline context stratum {key} has fewer than two sessions")
            context_cells[key] = cell
    if context_cells:
        expected_context = {
            (condition, persona, device, traffic)
            for condition in CONDITIONS
            for persona in PERSONAS
            for device in DEVICE_STRATA
            for traffic in TRAFFIC_STRATA
        }
        missing_context = sorted(expected_context - set(context_cells))
        if missing_context:
            raise ValueError(
                f"baseline is missing {len(missing_context)} context strata; "
                f"first missing: {missing_context[0]}"
            )
    return {
        "payload": payload,
        "cells": cells,
        "context_strata": context_cells,
        "path": str(path),
        "sha256": _sha256(path),
    }


def _baseline_mean_sd(
    cell: dict[str, Any], outcome: str
) -> tuple[float, float, int]:
    n = int(cell["n_sessions"])
    if outcome == "conversion":
        value = float(cell["conversion_rate"])
        # Sample SD of a Bernoulli outcome reconstructed from the exact cell
        # count. This supplies the simulator Monte Carlo component below.
        sd = math.sqrt(value * (1.0 - value) * n / (n - 1))
        return value, sd, n
    if outcome == "session_length_steps":
        return float(cell["step_count_mean"]), float(cell["step_count_sd"]), n
    if outcome == "funnel_depth":
        return float(cell["funnel_depth_mean"]), float(cell["funnel_depth_sd"]), n
    raise ValueError(f"unsupported fidelity outcome: {outcome}")


def _standardized_simulator_mean(
    baseline: dict[str, Any],
    condition: str,
    persona: str,
    human_records: Sequence[ParticipantRecord],
    outcome: str,
    *,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Standardize simulator means to the human exogenous-context mix."""
    context_cells = baseline.get("context_strata") or {}
    if not context_cells:
        sim_mean, sim_sd, sim_n = _baseline_mean_sd(
            baseline["cells"][(condition, persona)], outcome
        )
        sim_se = sim_sd / math.sqrt(sim_n)
        return {
            "mean": sim_mean,
            "draw": rng.gauss(sim_mean, sim_se) if rng else sim_mean,
            "mean_se": sim_se,
            "total_simulation_n": sim_n,
            "weights": None,
            "method": "precomputed aggregate simulator cell",
        }

    if not human_records:
        raise ValueError("cannot context-standardize without human records")
    counts = Counter(
        (record.device_type, record.traffic_source) for record in human_records
    )
    total = len(human_records)
    weighted_mean = 0.0
    weighted_draw = 0.0
    variance = 0.0
    total_simulation_n = 0
    weights: dict[str, float] = {}
    for (device, traffic), count in sorted(counts.items()):
        key = (condition, persona, device, traffic)
        if key not in context_cells:
            raise ValueError(f"simulator baseline missing observed context stratum: {key}")
        sim_mean, sim_sd, sim_n = _baseline_mean_sd(context_cells[key], outcome)
        weight = count / total
        sim_se = sim_sd / math.sqrt(sim_n)
        weighted_mean += weight * sim_mean
        weighted_draw += weight * (rng.gauss(sim_mean, sim_se) if rng else sim_mean)
        variance += (weight * sim_se) ** 2
        total_simulation_n += sim_n
        weights[f"{device}|{traffic}"] = weight
    return {
        "mean": weighted_mean,
        "draw": weighted_draw,
        "mean_se": math.sqrt(variance),
        "total_simulation_n": total_simulation_n,
        "weights": weights,
        "method": "standardized to recorded joint device/traffic distribution",
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        for index in order[position:end]:
            ranks[index] = average_rank
        position = end
    return ranks


def _spearman(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return float("nan")
    ranks_a = _average_ranks(values_a)
    ranks_b = _average_ranks(values_b)
    mean_a = mean(ranks_a)
    mean_b = mean(ranks_b)
    numerator = sum(
        (value_a - mean_a) * (value_b - mean_b)
        for value_a, value_b in zip(ranks_a, ranks_b)
    )
    denominator = math.sqrt(
        sum((value - mean_a) ** 2 for value in ranks_a)
        * sum((value - mean_b) ** 2 for value in ranks_b)
    )
    return numerator / denominator if denominator else float("nan")


def fidelity_analysis(
    records: Sequence[ParticipantRecord],
    baseline: dict[str, Any],
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Compare recorded-landing human sessions with all ten simulator cells."""
    eligible = [
        record
        for record in _valid_records(records)
        if record.recorded_landing
    ]
    human_cells: dict[tuple[str, str], list[ParticipantRecord]] = defaultdict(list)
    for record in eligible:
        human_cells[(record.condition, record.persona)].append(record)

    rng = random.Random(seed)
    endpoints: dict[str, Any] = {}
    for outcome, (_, _, margin) in FIDELITY_OUTCOMES.items():
        cell_rows: list[dict[str, Any]] = []
        missing_cells: list[str] = []
        for condition in CONDITIONS:
            for persona in PERSONAS:
                key = (condition, persona)
                cell_records = human_cells.get(key, [])
                values = [float(getattr(record, outcome)) for record in cell_records]
                if not cell_records:
                    missing_cells.append(f"{condition}/{persona}")
                    human_mean = float("nan")
                    sim_result = {
                        "mean": float("nan"),
                        "mean_se": float("nan"),
                        "total_simulation_n": 0,
                        "weights": None,
                        "method": "not estimable without a human cell",
                    }
                    gap = float("nan")
                else:
                    human_mean = mean(values)
                    sim_result = _standardized_simulator_mean(
                        baseline,
                        condition,
                        persona,
                        cell_records,
                        outcome,
                    )
                    gap = human_mean - float(sim_result["mean"])
                cell_rows.append({
                    "condition": condition,
                    "persona": persona,
                    "human_n": len(values),
                    "human_mean": human_mean,
                    "simulation_n": sim_result["total_simulation_n"],
                    "simulation_mean": sim_result["mean"],
                    "simulation_mean_se": sim_result["mean_se"],
                    "simulation_context_weights": sim_result["weights"],
                    "simulation_standardization": sim_result["method"],
                    "gap_human_minus_simulation": gap,
                })

        if missing_cells:
            endpoints[outcome] = {
                "margin": [-margin, margin],
                "analyzable": False,
                "missing_human_cells": missing_cells,
                "estimate": float("nan"),
                "bootstrap_ci90": [float("nan"), float("nan")],
                "bootstrap_ci95": [float("nan"), float("nan")],
                "equivalent": False,
                "cells": cell_rows,
            }
            continue

        estimate = mean([float(row["gap_human_minus_simulation"]) for row in cell_rows])
        bootstrap_estimates: list[float] = []
        for _ in range(bootstrap):
            gaps: list[float] = []
            for row in cell_rows:
                key = (str(row["condition"]), str(row["persona"]))
                source_records = human_cells[key]
                sampled_records = [
                    rng.choice(source_records) for _ in range(len(source_records))
                ]
                human_draw = mean(
                    [float(getattr(record, outcome)) for record in sampled_records]
                )
                sim_draw = _standardized_simulator_mean(
                    baseline,
                    key[0],
                    key[1],
                    sampled_records,
                    outcome,
                    rng=rng,
                )
                gaps.append(human_draw - float(sim_draw["draw"]))
            bootstrap_estimates.append(mean(gaps))
        ci90 = (
            _percentile(bootstrap_estimates, 0.05),
            _percentile(bootstrap_estimates, 0.95),
        )
        ci95 = (
            _percentile(bootstrap_estimates, 0.025),
            _percentile(bootstrap_estimates, 0.975),
        )
        simulator_mc_se = math.sqrt(
            sum(
                float(row["simulation_mean_se"]) ** 2
                for row in cell_rows
            )
        ) / len(cell_rows)
        human_persona_means = []
        simulation_persona_means = []
        for persona in PERSONAS:
            persona_rows = [row for row in cell_rows if row["persona"] == persona]
            human_persona_means.append(
                mean([float(row["human_mean"]) for row in persona_rows])
            )
            simulation_persona_means.append(
                mean([float(row["simulation_mean"]) for row in persona_rows])
            )
        endpoints[outcome] = {
            "margin": [-margin, margin],
            "analyzable": True,
            "missing_human_cells": [],
            "estimate": estimate,
            "bootstrap_ci90": [ci90[0], ci90[1]],
            "bootstrap_ci95": [ci95[0], ci95[1]],
            "equivalent": ci90[0] > -margin and ci90[1] < margin,
            "mean_absolute_cell_gap": mean(
                [abs(float(row["gap_human_minus_simulation"])) for row in cell_rows]
            ),
            "simulator_aggregate_mc_se": simulator_mc_se,
            "descriptive_persona_spearman": _spearman(
                human_persona_means, simulation_persona_means
            ),
            "cells": cell_rows,
        }

    return {
        "analysis_set": "Recorded-landing fidelity population",
        "n": len(eligible),
        "n_by_condition": dict(Counter(record.condition for record in eligible)),
        "n_by_persona": dict(Counter(record.persona for record in eligible)),
        "n_by_device": dict(Counter(record.device_type for record in eligible)),
        "n_by_traffic_source": dict(
            Counter(record.traffic_source for record in eligible)
        ),
        "n_by_device_and_traffic": {
            f"{device}|{traffic}": count
            for (device, traffic), count in sorted(
                Counter(
                    (record.device_type, record.traffic_source)
                    for record in eligible
                ).items()
            )
        },
        "simulation_uncertainty": (
            "Parametric simulator-mean resampling combined with participant-level "
            "stratified bootstrap; context-stratified references are standardized "
            "to each resampled cell's joint device/traffic distribution"
        ),
        "context_standardized": bool(baseline.get("context_strata")),
        "endpoints": endpoints,
        "all_endpoints_equivalent": bool(endpoints)
        and all(value["analyzable"] and value["equivalent"] for value in endpoints.values()),
    }


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return (max(0.0, centre - half), min(1.0, centre + half))


def _progression_from_depths(depths: Sequence[int]) -> dict[str, dict[str, float | int]]:
    """Conditional funnel-advance counts per adjacent stage pair.

    Mirrors ``_funnel_progression`` in ``evaluate_study_policy_baseline.py``:
    reached stage k means max funnel depth >= k, so the proportions on the
    human and simulator sides are the same function of the same observable.
    """
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


def _pooled_simulator_progression(
    baseline: dict[str, Any], persona: str
) -> dict[str, dict[str, float | int]] | None:
    """Persona progression from the reference, pooled over the two policies.

    Pooling sums the two cells' funnel-depth histograms (the cells carry equal
    session counts by construction), which matches how the human side pools
    conditions within a persona. Returns ``None`` when the baseline predates
    the histogram emission, in which case the ordering analysis reports itself
    non-analyzable instead of silently inventing a prediction.
    """
    pooled: Counter[int] = Counter()
    for condition in CONDITIONS:
        cell = baseline["cells"].get((condition, persona))
        if not cell or not isinstance(cell.get("funnel_depth_histogram"), dict):
            return None
        for depth, count in cell["funnel_depth_histogram"].items():
            pooled[int(depth)] += int(count)
    depths = [depth for depth, count in pooled.items() for _ in range(count)]
    return _progression_from_depths(depths)


def transition_ordering_analysis(
    records: Sequence[ParticipantRecord],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Directional test of the simulator's predicted funnel-advance ordering.

    For every stage pair and every pair of personas present in the data, the
    human proportions (pooled over the two policy conditions, which is
    legitimate because the ordering claim is about personas, not policies) are
    compared in the direction the locked simulator reference predicts. The
    prediction is treated as fixed: simulator Monte Carlo error at 2,000
    sessions per cell is negligible against human cells of tens.

    One-sided p-values use the unpooled two-proportion z statistic; interval
    estimates are Newcombe hybrid-Wilson. Contrasts whose conditioning set is
    thinner than ``MIN_REACHED_FOR_ORDERING_TEST`` on either side are shown
    but not tested.
    """
    eligible = [record for record in _valid_records(records) if record.recorded_landing]
    persona_depths: dict[str, list[int]] = defaultdict(list)
    persona_condition_depths: dict[tuple[str, str], list[int]] = defaultdict(list)
    for record in eligible:
        persona_depths[record.persona].append(record.funnel_depth)
        persona_condition_depths[(record.persona, record.condition)].append(record.funnel_depth)
    personas_present = [persona for persona in PERSONAS if persona_depths.get(persona)]

    human: dict[str, Any] = {}
    simulator: dict[str, Any] = {}
    baseline_has_histograms = True
    for persona in personas_present:
        progression = _progression_from_depths(persona_depths[persona])
        for values in progression.values():
            values["wilson_ci95"] = list(
                _wilson_ci(int(values["advanced"]), int(values["reached"]))
            )
        human[persona] = {
            "n": len(persona_depths[persona]),
            "n_by_condition": {
                condition: len(persona_condition_depths.get((persona, condition), []))
                for condition in CONDITIONS
            },
            "progression": progression,
            "progression_by_condition": {
                condition: _progression_from_depths(
                    persona_condition_depths.get((persona, condition), [])
                )
                for condition in CONDITIONS
            },
        }
        sim_progression = _pooled_simulator_progression(baseline, persona)
        if sim_progression is None:
            baseline_has_histograms = False
        else:
            simulator[persona] = sim_progression

    contrasts: list[dict[str, Any]] = []
    confirmatory_keys = set(CONFIRMATORY_ORDERING_CONTRASTS)
    if baseline_has_histograms:
        for source, target in FUNNEL_PROGRESSION_PAIRS:
            pair_key = f"{source}_to_{target}"
            for index_a, persona_a in enumerate(personas_present):
                for persona_b in personas_present[index_a + 1:]:
                    sim_a = float(simulator[persona_a][pair_key]["proportion"])
                    sim_b = float(simulator[persona_b][pair_key]["proportion"])
                    if math.isnan(sim_a) or math.isnan(sim_b) or sim_a == sim_b:
                        continue
                    # Orient the contrast so "higher" is the persona the
                    # reference predicts to advance more often.
                    higher, lower = (
                        (persona_a, persona_b) if sim_a > sim_b else (persona_b, persona_a)
                    )
                    high_row = human[higher]["progression"][pair_key]
                    low_row = human[lower]["progression"][pair_key]
                    n_high, k_high = int(high_row["reached"]), int(high_row["advanced"])
                    n_low, k_low = int(low_row["reached"]), int(low_row["advanced"])
                    estimable = (
                        n_high >= MIN_REACHED_FOR_ORDERING_TEST
                        and n_low >= MIN_REACHED_FOR_ORDERING_TEST
                    )
                    entry: dict[str, Any] = {
                        "stage_pair": pair_key,
                        "higher": higher,
                        "lower": lower,
                        "confirmatory": (pair_key, higher, lower) in confirmatory_keys,
                        "simulator_higher_proportion": float(
                            simulator[higher][pair_key]["proportion"]
                        ),
                        "simulator_lower_proportion": float(
                            simulator[lower][pair_key]["proportion"]
                        ),
                        "simulator_gap": abs(sim_a - sim_b),
                        "human_higher_reached": n_high,
                        "human_higher_proportion": k_high / n_high if n_high else float("nan"),
                        "human_lower_reached": n_low,
                        "human_lower_proportion": k_low / n_low if n_low else float("nan"),
                        "estimable": estimable,
                    }
                    if estimable:
                        p_high, p_low = k_high / n_high, k_low / n_low
                        diff = p_high - p_low
                        se = math.sqrt(
                            p_high * (1.0 - p_high) / n_high
                            + p_low * (1.0 - p_low) / n_low
                        )
                        l_high, u_high = _wilson_ci(k_high, n_high)
                        l_low, u_low = _wilson_ci(k_low, n_low)
                        entry.update({
                            "human_gap": diff,
                            "newcombe_ci95": [
                                diff - math.sqrt((p_high - l_high) ** 2 + (u_low - p_low) ** 2),
                                diff + math.sqrt((u_high - p_high) ** 2 + (p_low - l_low) ** 2),
                            ],
                            "one_sided_p": (
                                1.0 - _normal_cdf(diff / se) if se > 0 else float("nan")
                            ),
                            "direction_matches": diff > 0,
                        })
                        entry["holds"] = bool(
                            se > 0
                            and entry["one_sided_p"] < ORDERING_ALPHA_ONE_SIDED
                        )
                    contrasts.append(entry)

    confirmatory = [entry for entry in contrasts if entry["confirmatory"]]
    analyzable = (
        baseline_has_histograms
        and len(confirmatory) == len(CONFIRMATORY_ORDERING_CONTRASTS)
        and all(entry["estimable"] for entry in confirmatory)
    )
    return {
        "analysis_set": "Recorded-landing transition-ordering population",
        "n": len(eligible),
        "estimand": (
            "Direction of persona differences in conditional funnel-advance "
            "proportions, pooled over policy conditions, against the ordering "
            "predicted by the locked simulator reference"
        ),
        "decision_rule": (
            "Intersection-union over the confirmatory contrasts at one-sided "
            f"alpha = {ORDERING_ALPHA_ONE_SIDED}; descriptive contrasts carry "
            "no confirmatory weight and a null there is uninformative"
        ),
        "baseline_has_histograms": baseline_has_histograms,
        "personas_present": personas_present,
        "human": human,
        "simulator_pooled": simulator,
        "contrasts": contrasts,
        "confirmatory_expected": [list(item) for item in CONFIRMATORY_ORDERING_CONTRASTS],
        "analyzable": analyzable,
        "all_confirmatory_hold": bool(
            analyzable and confirmatory
            and all(entry.get("holds") for entry in confirmatory)
        ),
    }


def stratified_bootstrap_ci(
    records: Sequence[ParticipantRecord],
    outcome: str,
    *,
    replicates: int,
    rng: random.Random,
) -> tuple[float, float]:
    cells: dict[tuple[str, str], list[ParticipantRecord]] = defaultdict(list)
    for record in records:
        cells[(record.condition, record.persona)].append(record)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled: list[ParticipantRecord] = []
        for values in cells.values():
            sampled.extend(rng.choice(values) for _ in range(len(values)))
        estimate = adjusted_difference(sampled, outcome)
        if not math.isnan(estimate):
            estimates.append(estimate)
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def stratified_randomization_pvalue(
    records: Sequence[ParticipantRecord],
    outcome: str,
    *,
    permutations: int,
    rng: random.Random,
) -> float:
    observed = adjusted_difference(records, outcome)
    if math.isnan(observed):
        return float("nan")
    by_persona: dict[str, list[ParticipantRecord]] = defaultdict(list)
    for record in records:
        by_persona[record.persona].append(record)
    exceed = 0
    completed = 0
    for _ in range(permutations):
        persona_effects: list[float] = []
        for persona in PERSONAS:
            values = by_persona.get(persona, [])
            n_v3 = sum(r.condition == "v3" for r in values)
            n_v2 = len(values) - n_v3
            if not n_v2 or not n_v3:
                continue
            shuffled = [float(getattr(r, outcome)) for r in values]
            rng.shuffle(shuffled)
            persona_effects.append(mean(shuffled[:n_v3]) - mean(shuffled[n_v3:]))
        if not persona_effects:
            continue
        permuted = mean(persona_effects)
        completed += 1
        if abs(permuted) >= abs(observed) - 1e-15:
            exceed += 1
    return (exceed + 1) / (completed + 1) if completed else float("nan")


def _cell_summary(records: Sequence[ParticipantRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for persona in PERSONAS:
            cell = [r for r in records if r.condition == condition and r.persona == persona]
            durations = [r.duration_s for r in cell if math.isfinite(r.duration_s)]
            rows.append({
                "condition": condition,
                "persona": persona,
                "n": len(cell),
                "mean_session_reward": mean([r.session_reward for r in cell]) if cell else None,
                "conversion_rate": mean([r.conversion for r in cell]) if cell else None,
                "mean_steps": mean([r.session_length_steps for r in cell]) if cell else None,
                "mean_funnel_depth": mean([r.funnel_depth for r in cell]) if cell else None,
                "mean_duration_s": mean(durations) if durations else None,
                "recorded_landing_rate": mean([int(r.recorded_landing) for r in cell]) if cell else None,
                "quality_pass_rate": mean([int(r.quality_ok) for r in cell]) if cell else None,
                "mean_decision_errors": mean([r.decision_errors for r in cell]) if cell else None,
                "mean_fallback_rate": mean([r.fallback_rate for r in cell]) if cell else None,
            })
    return rows


def analyze(
    records: Sequence[ParticipantRecord],
    *,
    bootstrap: int,
    permutations: int,
    seed: int,
    label: str,
) -> dict[str, Any]:
    eligible = _valid_records(records)
    rng_boot = random.Random(seed)
    rng_perm = random.Random(seed + 1)
    outcomes: dict[str, Any] = {}
    for outcome in ("session_reward", "conversion"):
        estimate = adjusted_difference(eligible, outcome)
        ci = stratified_bootstrap_ci(
            eligible, outcome, replicates=bootstrap, rng=rng_boot
        )
        robust_ci = robust_difference_ci(eligible, outcome)
        pvalue = stratified_randomization_pvalue(
            eligible, outcome, permutations=permutations, rng=rng_perm
        )
        outcomes[outcome] = {
            "estimand": "equal-persona average of within-persona V3 minus V2",
            "estimate": estimate,
            "heteroskedastic_robust_ci95": [robust_ci[0], robust_ci[1]],
            "bootstrap_ci95": [ci[0], ci[1]],
            "stratified_randomization_p_two_sided": pvalue,
        }
    return {
        "analysis_set": label,
        "n": len(eligible),
        "n_by_condition": dict(Counter(r.condition for r in eligible)),
        "n_by_persona": dict(Counter(r.persona for r in eligible)),
        "outcomes": outcomes,
        "cells": _cell_summary(eligible),
    }


def _write_csv(path: Path, records: Sequence[ParticipantRecord]) -> None:
    fieldnames = list(ParticipantRecord.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.4f}"
    return str(value)


def _json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with JSON ``null``."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _report(results: dict[str, Any]) -> str:
    lines = [
        "# Clickworker frozen-policy analysis",
        "",
        "The main analysis is participant-level intention-to-treat. Quality-filtered results are a sensitivity analysis because duration, navigation, dwell and dismissals can be affected by policy.",
        "",
    ]
    cutoff = results.get("cutoff")
    if cutoff:
        lines.extend([
            f"Recruitment-window cutoff: data from {cutoff['exclusive_utc_bound']} UTC "
            f"onward excluded ({cutoff['dropped_assignments']} assignments, "
            f"{cutoff['dropped_sessions']} sessions dropped).",
            "",
        ])
    ordering = results["transition_ordering"]
    lines.extend([
        "## Transition-ordering endpoint",
        "",
        f"Recorded-landing N = {ordering['n']}",
        ordering["estimand"] + ".",
        "",
    ])
    if not ordering["baseline_has_histograms"]:
        lines.extend([
            "NOT ANALYZABLE: the simulator baseline predates the funnel-depth "
            "histogram emission; regenerate the reference with the current "
            "evaluate_study_policy_baseline.py.",
            "",
        ])
    else:
        lines.extend([
            "| Contrast | Stage | Sim gap | Human higher | Human lower | Human gap | Newcombe 95% CI | One-sided p | Holds |",
            "|---|---|---:|---:|---:|---:|---:|---:|:---:|",
        ])
        ordered_rows = sorted(
            ordering["contrasts"], key=lambda row: not row["confirmatory"]
        )
        for row in ordered_rows:
            label = f"{row['higher']} > {row['lower']}"
            if row["confirmatory"]:
                label = f"**{label}**"
            if not row["estimable"]:
                verdict = "not tested"
                ci_text = p_text = gap_text = "NA"
            else:
                verdict = "yes" if row.get("holds") else "no"
                ci = row["newcombe_ci95"]
                ci_text = f"[{_fmt(ci[0])}, {_fmt(ci[1])}]"
                p_text = _fmt(row.get("one_sided_p"))
                gap_text = _fmt(row.get("human_gap"))
            lines.append(
                f"| {label} | {row['stage_pair']} | {_fmt(row['simulator_gap'])} | "
                f"{_fmt(row['human_higher_proportion'])} (n={row['human_higher_reached']}) | "
                f"{_fmt(row['human_lower_proportion'])} (n={row['human_lower_reached']}) | "
                f"{gap_text} | {ci_text} | {p_text} | {verdict} |"
            )
        lines.extend([
            "",
            "Bold contrasts are confirmatory (intersection-union, one-sided "
            f"alpha = {ORDERING_ALPHA_ONE_SIDED}); the rest are descriptive and "
            "a null there is uninformative. The confirmatory ordering claim: "
            + (
                "HOLDS."
                if ordering["all_confirmatory_hold"]
                else ("NOT ESTABLISHED." if ordering["analyzable"] else "NOT ANALYZABLE.")
            ),
            "",
        ])
    fidelity = results["primary_fidelity"]
    lines.extend([
        "## Primary simulator fidelity",
        "",
        f"Recorded-landing N = {fidelity['n']}",
        (
            "Simulator means are standardized to the recorded joint device/traffic "
            "distribution within each randomized cell."
            if fidelity.get("context_standardized")
            else "Simulator means use the baseline's aggregate context distribution."
        ),
        "",
        "| Outcome | Human-simulation gap | Bootstrap 90% CI | Margin | Equivalent | Mean absolute cell gap | Persona Spearman |",
        "|---|---:|---:|---:|:---:|---:|---:|",
    ])
    for name, values in fidelity["endpoints"].items():
        ci = values["bootstrap_ci90"]
        margin = values["margin"]
        lines.append(
            f"| {name} | {_fmt(values['estimate'])} | [{_fmt(ci[0])}, {_fmt(ci[1])}] | "
            f"[{_fmt(margin[0])}, {_fmt(margin[1])}] | {'yes' if values['equivalent'] else 'no'} | "
            f"{_fmt(values.get('mean_absolute_cell_gap'))} | "
            f"{_fmt(values.get('descriptive_persona_spearman'))} |"
        )
    lines.extend([
        "",
        "The global calibration claim requires equivalence for all three endpoints: "
        + ("PASS." if fidelity["all_endpoints_equivalent"] else "NOT ESTABLISHED."),
        "",
    ])
    for key in ("itt", "recorded_landing_policy_sensitivity", "quality_sensitivity"):
        analysis = results[key]
        lines.extend([f"## {analysis['analysis_set']}", "", f"N = {analysis['n']}", ""])
        lines.extend([
            "| Outcome | V3-V2 estimate | Robust 95% CI | Bootstrap 95% CI | Randomization p |",
            "|---|---:|---:|---:|---:|",
        ])
        for name, values in analysis["outcomes"].items():
            ci = values["bootstrap_ci95"]
            robust_ci = values["heteroskedastic_robust_ci95"]
            lines.append(
                f"| {name} | {_fmt(values['estimate'])} | [{_fmt(robust_ci[0])}, {_fmt(robust_ci[1])}] | "
                f"[{_fmt(ci[0])}, {_fmt(ci[1])}] | {_fmt(values['stratified_randomization_p_two_sided'])} |"
            )
        lines.extend(["", "Cell estimates are descriptive; policy-by-persona interactions are exploratory.", ""])
    lines.extend([
        "## Audit",
        "",
        f"- Randomized assignments represented: {results['audit_summary']['assigned_records']}",
        f"- Assignments without an observed shop session: {results['audit_summary']['unobserved_assignments']}",
        f"- Participants with duplicate observed sessions: {results['audit_summary']['participants_with_duplicates']}",
        f"- Logged decision-request errors: {results['audit_summary']['decision_request_errors']}",
        f"- Unattributed or mismatched sessions: {results['audit_summary']['audit_events']}",
        "- Action-level IPS/SNIPS/DR is intentionally not run because deterministic policy logs do not establish overlap.",
        "",
    ])
    return "\n".join(lines)


def parse_db_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip().lower()
    if label not in CONDITIONS:
        raise argparse.ArgumentTypeError("LABEL must be v2 or v3")
    path = Path(raw_path).expanduser()
    return label, path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", action="append", required=True, type=parse_db_spec, metavar="LABEL=PATH",
        help="shop SQLite database; pass once for v2 and once for v3",
    )
    parser.add_argument("--dispatcher-db", type=Path, help="optional assignment-ledger SQLite database")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(
            "Experiments/study_policy_baselines/"
            "clickworker_pre_recruitment_20260721/policy_archetype_baseline.json"
        ),
        help="policy-matched 2 x 5 simulator reference JSON",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("Experiments/clickworker_analysis"))
    parser.add_argument(
        "--cutoff",
        type=parse_cutoff,
        default=None,
        help=(
            "drop assignments made, and sessions first seen, after this UTC "
            "bound (e.g. the close of the preregistered recruitment window). "
            "A bare YYYY-MM-DD keeps that whole day. Default: keep everything."
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--personas",
        default="",
        help=(
            "comma-separated recruited personas to restrict every cell-averaged "
            "estimand to (e.g. the recruited subset of a design that assigns "
            "fewer personas than the reference evaluates). Empty means all "
            "personas."
        ),
    )
    args = parser.parse_args(argv)

    global PERSONAS
    requested = [item.strip() for item in args.personas.split(",") if item.strip()]
    if requested:
        unknown = sorted(set(requested) - set(ALL_PERSONAS))
        if unknown:
            parser.error(
                f"personas not recognized: {unknown}; available: {list(ALL_PERSONAS)}"
            )
        PERSONAS = tuple(item for item in ALL_PERSONAS if item in requested)
    else:
        PERSONAS = ALL_PERSONAS

    specs = dict(args.db)
    if set(specs) != set(CONDITIONS):
        parser.error("provide exactly one --db for v2 and one for v3")
    for path in specs.values():
        if not path.exists():
            parser.error(f"database not found: {path}")
    if args.dispatcher_db and not args.dispatcher_db.exists():
        parser.error(f"dispatcher database not found: {args.dispatcher_db}")
    if not args.baseline.exists():
        parser.error(f"simulator baseline not found: {args.baseline}")
    if args.bootstrap < 1 or args.permutations < 1:
        parser.error("--bootstrap and --permutations must be positive")

    sessions: list[RawSession] = []
    for condition in CONDITIONS:
        sessions.extend(load_shop_sessions(condition, specs[condition]))
    assignments = load_assignments(args.dispatcher_db) if args.dispatcher_db else None

    dropped_sessions = 0
    dropped_assignments = 0
    if args.cutoff is not None:
        sessions, assignments, dropped_sessions, dropped_assignments = apply_cutoff(
            sessions, assignments, args.cutoff
        )

    records, audit = build_participant_records(sessions, assignments)
    # When an assignment ledger is available, observed but unrandomized rows
    # are audit records rather than members of the randomized population.
    itt = [
        record
        for record in _valid_records(records)
        if assignments is None or record.assigned
    ]
    quality = [record for record in itt if record.quality_ok]
    attempted = [record for record in itt if record.recorded_landing]
    baseline = load_policy_baseline(args.baseline)
    results = {
        "schema_version": 1,
        "estimand": "equal-persona participant-level V3 minus V2 fixed-policy effect",
        "policy_serving": "deterministic_frozen",
        "personas": list(PERSONAS),
        "cutoff": (
            {
                "exclusive_utc_bound": args.cutoff.isoformat(),
                "dropped_assignments": dropped_assignments,
                "dropped_sessions": dropped_sessions,
            }
            if args.cutoff is not None else None
        ),
        "seed": args.seed,
        "bootstrap_replicates": args.bootstrap,
        "randomization_permutations": args.permutations,
        "inputs": {
            "shop_databases": {
                key: {"path": str(value), "sha256": _sha256(value)}
                for key, value in specs.items()
            },
            "dispatcher_database": (
                {"path": str(args.dispatcher_db), "sha256": _sha256(args.dispatcher_db)}
                if args.dispatcher_db else None
            ),
            "simulator_baseline": {
                "path": baseline["path"],
                "sha256": baseline["sha256"],
                "study_id": baseline["payload"].get("study_id"),
            },
        },
        "transition_ordering": transition_ordering_analysis(attempted, baseline),
        "primary_fidelity": fidelity_analysis(
            attempted,
            baseline,
            bootstrap=args.bootstrap,
            seed=args.seed + 20,
        ),
        "itt": analyze(
            itt, bootstrap=args.bootstrap, permutations=args.permutations, seed=args.seed, label="Intention-to-treat"
        ),
        "quality_sensitivity": analyze(
            quality, bootstrap=args.bootstrap, permutations=args.permutations, seed=args.seed + 10, label="Quality-filtered sensitivity"
        ),
        "recorded_landing_policy_sensitivity": analyze(
            attempted,
            bootstrap=args.bootstrap,
            permutations=args.permutations,
            seed=args.seed + 30,
            label="Recorded-landing policy sensitivity",
        ),
        "audit_summary": {
            "assigned_records": sum(record.assigned for record in records),
            "unobserved_assignments": sum(record.assigned and not record.observed_session for record in records),
            "participants_with_duplicates": sum(record.duplicate_sessions > 0 for record in records),
            "decision_request_errors": sum(record.decision_errors for record in records),
            "audit_events": len(audit),
        },
        "audit_events": audit,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "session_level.csv", records)
    (args.out_dir / "analysis_results.json").write_text(
        json.dumps(_json_safe(results), indent=2, allow_nan=False), encoding="utf-8"
    )
    (args.out_dir / "analysis_report.md").write_text(_report(results), encoding="utf-8")
    print(f"Wrote participant-level analysis to {args.out_dir}")
    print(
        "ITT reward V3-V2:",
        _fmt(results["itt"]["outcomes"]["session_reward"]["estimate"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
