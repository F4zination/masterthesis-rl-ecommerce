import random
import json
from pathlib import Path
from threading import Lock
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from ..models import BanditArmStat, DecisionLog
from ..config import (
    POLICY_MODE,
    OFFLINE_POLICY_PATH,
    ACTIVE_POLICY_POINTER_PATH,
    TIMING_ENABLED,
    MAX_OPPORTUNITIES_PER_SESSION,
    MIN_DECISION_COOLDOWN_MS,
    OFFLINE_POLICY_DEFER_MS_DEFAULT,
    FREEZE_POLICY,
)
from .ppo import get_ppo_health_status, run_ppo_policy

from shared_schema.constants import (
    EPSILON,
    PRIOR_COUNT,
    PRIOR_MEAN,
    LOOKBACK_MINUTES,
    ACTION_COST,
    CONTEXT_SCHEMA_VERSION,
    DECISION_POINTS,
)
from shared_schema.features import (
    _normalize_context,
    _eligible_actions,
    _decision_point_from_page,
    _event_reward,
    _history_from_actions,
)

_POLICY_CACHE_LOCK = Lock()
_POLICY_CACHE_MTIME: float | None = None
_POLICY_CACHE_PAYLOAD: dict | None = None
_POLICY_CACHE_PATH: str | None = None




def _estimate_mean(stat: BanditArmStat | None) -> float:
    """Compute the Bayesian-smoothed mean reward estimate for a bandit arm.

    Applies an additive prior of ``PRIOR_COUNT`` pseudo-observations each
    contributing ``PRIOR_MEAN``, which shrinks arms with few real impressions
    towards the prior and avoids cold-start over-exploitation.

    Args:
        stat: The ``BanditArmStat`` row for this arm, or ``None`` if the arm
            has not yet been created.

    Returns:
        Smoothed mean reward estimate as a float.
    """
    if stat is None:
        return PRIOR_MEAN
    return (stat.reward_sum + PRIOR_COUNT * PRIOR_MEAN) / (stat.impressions + PRIOR_COUNT)



def _get_or_create_stat(
    db: Session,
    decision_point: str,
    context_key: str,
    action: str,
) -> BanditArmStat | None:
    """Fetch or create the ``BanditArmStat`` row for a specific bandit arm.

    If no row exists yet for the ``(decision_point, context_key, action)``
    triplet, a new one is created with zero impressions and reward, flushed
    to the session (but not yet committed).

    Args:
        db: Active SQLAlchemy database session.
        decision_point: The funnel step identifier.
        context_key: Pipe-delimited discrete context string.
        action: The widget action name.

    Returns:
        The existing or newly created ``BanditArmStat`` instance. Under a
        frozen policy, returns ``None`` for a missing arm instead of mutating
        the release database.
    """
    stat = (
        db.query(BanditArmStat)
        .filter(
            BanditArmStat.decision_point == decision_point,
            BanditArmStat.context_key == context_key,
            BanditArmStat.action == action,
        )
        .first()
    )
    if stat:
        return stat

    # Frozen study serving must keep the policy tables read-only. PPO-only mode does not use these
    # estimates, and development fallbacks can still interpret a missing row
    # through _estimate_mean(None) without mutating the locked bandit parameters.
    if FREEZE_POLICY:
        return None

    stat = BanditArmStat(
        decision_point=decision_point,
        context_key=context_key,
        action=action,
        impressions=0,
        reward_sum=0.0,
        updated_at=datetime.utcnow(),
    )
    db.add(stat)
    db.flush()
    return stat


def _state_key(state: dict) -> str:
    return json.dumps(state, sort_keys=True, ensure_ascii=True)


def _resolve_policy_path() -> Path:
    default_path = Path(OFFLINE_POLICY_PATH)
    pointer = Path(ACTIVE_POLICY_POINTER_PATH)

    if not pointer.exists():
        return default_path

    try:
        version = pointer.read_text(encoding="utf-8").strip()
    except OSError:
        return default_path

    if not version:
        return default_path

    candidate = pointer.parent / version / "trained_policy.json"
    return candidate if candidate.exists() else default_path


def _load_offline_policy() -> dict | None:
    global _POLICY_CACHE_MTIME, _POLICY_CACHE_PAYLOAD, _POLICY_CACHE_PATH

    policy_path = _resolve_policy_path()
    if not policy_path.exists():
        return None

    try:
        mtime = policy_path.stat().st_mtime
    except OSError:
        return None

    with _POLICY_CACHE_LOCK:
        if (
            _POLICY_CACHE_PAYLOAD is not None
            and _POLICY_CACHE_MTIME == mtime
            and _POLICY_CACHE_PATH == str(policy_path)
        ):
            return _POLICY_CACHE_PAYLOAD

        try:
            payload = json.loads(policy_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        if not isinstance(payload, dict):
            return None

        _POLICY_CACHE_PAYLOAD = payload
        _POLICY_CACHE_MTIME = mtime
        _POLICY_CACHE_PATH = str(policy_path)
        return payload


def get_policy_health_status() -> dict:
    """Return health diagnostics for startup checks and observability.

    This does not raise: callers can use the returned status to decide
    whether runtime will use offline policy directly or fallback behavior.
    """
    status = {
        "policy_mode": POLICY_MODE,
        "policy_path": str(_resolve_policy_path()),
        "policy_pointer_path": ACTIVE_POLICY_POINTER_PATH,
        "timing_enabled": TIMING_ENABLED,
        "max_opportunities_per_session": MAX_OPPORTUNITIES_PER_SESSION,
        "min_decision_cooldown_ms": MIN_DECISION_COOLDOWN_MS,
        "offline_policy_available": False,
        "offline_policy_valid": False,
        "ppo_checkpoint_available": False,
        "ppo_checkpoint_valid": False,
        "fallback": "noop" if POLICY_MODE == "ppo_only" else "bandit",
        "reason": "",
    }

    ppo_status = get_ppo_health_status()
    status["ppo_checkpoint_path"] = ppo_status.get("checkpoint_path")
    status["ppo_checkpoint_available"] = bool(ppo_status.get("available"))
    status["ppo_checkpoint_valid"] = bool(ppo_status.get("valid"))
    status["ppo_checkpoint_study_contract_valid"] = bool(
        ppo_status.get("study_contract_valid")
    )
    status["ppo_checkpoint_study_contract_error"] = ppo_status.get(
        "study_contract_error", ""
    )
    if ppo_status.get("summary"):
        status["ppo_summary"] = ppo_status.get("summary")

    payload = _load_offline_policy()
    if payload is None:
        status["reason"] = "offline policy file missing or unreadable"
        if POLICY_MODE in {"offline_only", "ppo_only"}:
            status["fallback"] = "noop"
        return status

    status["offline_policy_available"] = True

    policy_map = payload.get("policy")
    defaults = payload.get("default_action_by_decision_point")
    if not isinstance(policy_map, dict) or not isinstance(defaults, dict):
        status["reason"] = "offline policy payload missing required keys"
        if POLICY_MODE == "offline_only":
            status["fallback"] = "noop"
        return status

    status["offline_policy_valid"] = True
    status["reason"] = "offline policy ready"
    status["policy_states"] = len(policy_map)
    status["default_points"] = len(defaults)

    if POLICY_MODE in {"offline_only", "ppo_only"}:
        status["fallback"] = "noop"
    elif POLICY_MODE == "offline_first":
        status["fallback"] = "bandit"
    else:
        status["fallback"] = "bandit"

    return status


def _run_bandit_policy(
    db: Session,
    decision_point: str,
    context_key: str,
    actions: list[str],
) -> tuple[str, float, str, dict]:
    stats_by_action = {
        action: _get_or_create_stat(db, decision_point, context_key, action)
        for action in actions
    }
    estimates = {action: _estimate_mean(stats_by_action[action]) for action in actions}
    max_value = max(estimates.values())
    best_actions = [a for a, v in estimates.items() if v == max_value]

    if FREEZE_POLICY:
        # Frozen deployments (Clickworker study) serve deterministic greedy:
        # no exploration, stable tie-break, degenerate propensity 1.0.
        action = best_actions[0]
        return action, 1.0, "epsilon_greedy_v2_frozen_greedy", {
            "epsilon": 0.0, "estimates": estimates,
        }

    is_explore = random.random() < EPSILON
    if is_explore:
        action = random.choice(actions)
    else:
        action = random.choice(best_actions)

    # Logged propensity is the *marginal* probability of the served action
    # under the full epsilon-greedy distribution — an explored action that
    # coincides with a greedy action also carries the greedy share.
    greedy_share = (1.0 - EPSILON) / len(best_actions) if action in best_actions else 0.0
    propensity = EPSILON / len(actions) + greedy_share

    return action, propensity, "epsilon_greedy_v2", {"epsilon": EPSILON, "estimates": estimates}


def _run_offline_policy(
    decision_point: str,
    normalized_context: dict,
    actions: list[str],
) -> tuple[str, float, str, dict] | None:
    payload = _load_offline_policy()
    if not payload:
        return None

    policy_map = payload.get("policy") or {}
    defaults = payload.get("default_action_by_decision_point") or {}
    if not isinstance(policy_map, dict) or not isinstance(defaults, dict):
        return None

    state_key = _state_key(normalized_context)
    action = policy_map.get(state_key)
    match_type = "state_match"

    if action is None:
        action = defaults.get(decision_point)
        match_type = "point_default"

    if action not in actions:
        return None

    algo = str(payload.get("algorithm") or "offline_tabular_policy")
    return action, 1.0, algo, {"match_type": match_type, "state_key": state_key}


def _session_history(db: Session, session_id: str) -> dict:
    """Reconstruct raw trajectory-history values from this session's decisions.

    Feeds the session's logged actions through the shared
    ``_history_from_actions`` recurrence (identical to the simulator's
    ``SimState`` bookkeeping), so the four ``HISTORY_FEATURES`` buckets the
    PPO policy conditions on mean the same thing at serving time as they did
    in training.

    Primed credit is an observed action-history feature regardless of whether
    the delayed-reward mechanism affects purchase probability. The simulator
    applies this same recurrence, and the explicit PPO vocabulary reserves
    every bucket even if one is absent from a finite seed dataset.
    """
    prior_actions = [
        row[0]
        for row in (
            db.query(DecisionLog.action)
            .filter(DecisionLog.session_id == session_id)
            .order_by(DecisionLog.timestamp.asc(), DecisionLog.id.asc())
            .all()
        )
    ]
    return _history_from_actions(prior_actions)


def _timing_gate(
    db: Session,
    session_id: str,
    context: dict,
) -> tuple[bool, int, str]:
    if not TIMING_ENABLED:
        return True, 0, "timing_disabled"

    if MAX_OPPORTUNITIES_PER_SESSION > 0:
        seen = db.query(DecisionLog).filter(DecisionLog.session_id == session_id).count()
        if seen >= MAX_OPPORTUNITIES_PER_SESSION:
            return False, 0, "session_cap_reached"

    if MIN_DECISION_COOLDOWN_MS > 0:
        last = (
            db.query(DecisionLog)
            .filter(DecisionLog.session_id == session_id)
            .order_by(DecisionLog.timestamp.desc())
            .first()
        )
        if last:
            elapsed_ms = int((datetime.utcnow() - last.timestamp).total_seconds() * 1000)
            if elapsed_ms < MIN_DECISION_COOLDOWN_MS:
                return False, MIN_DECISION_COOLDOWN_MS - elapsed_ms, "cooldown"

    elapsed_ms = int(context.get("elapsed_ms") or 0)
    next_earliest_ms = int(context.get("next_earliest_ms") or 0)
    if next_earliest_ms > 0 and elapsed_ms < next_earliest_ms:
        return False, next_earliest_ms - elapsed_ms, "not_yet_eligible"

    return True, 0, "eligible"


def get_decision(
    db: Session,
    session_id: str,
    decision_point: str,
    context: dict | None = None,
    worker_id: str = "",
    persona: str = "",
) -> dict:
    """Return an action with optional timing gate and offline-policy-first fallback chain.

    ``worker_id`` and ``persona`` are Clickworker study attribution values
    stamped into the decision log's ``context_json``; they do not affect the
    policy or the bandit context key.
    """
    if decision_point not in DECISION_POINTS:
        raise ValueError(f"unsupported decision_point: {decision_point!r}")

    context = dict(context or {})
    actions = _eligible_actions(decision_point)

    should_show, next_check_after_ms, gate_reason = _timing_gate(db, session_id, context)
    if not should_show:
        # Gated opportunities never serve an action and are deliberately not
        # written to DecisionLog (they would consume the opportunity cap and
        # distort session_step); they remain observable through the frontend's
        # opportunity_checked / opportunity_skipped event stream.
        return {
            "decision_id": None,
            "action": "no-op",
            "propensity": 1.0,
            "decision_point": decision_point,
            "model": "timing_gate_v1",
            "should_show": False,
            "next_check_after_ms": next_check_after_ms or OFFLINE_POLICY_DEFER_MS_DEFAULT,
            "reason_code": gate_reason,
        }

    # Inject reconstructed trajectory history so the sequential policy (PPO,
    # and FQI artifacts trained with sequential=true) observes the same
    # HISTORY_FEATURES buckets it was trained on.  The raw values land in
    # context_raw, the buckets in context_normalized; the bandit's context_key
    # stays myopic (POINT_FEATURES only).
    context.update(_session_history(db, session_id))
    context_key, normalized_context = _normalize_context(
        decision_point, context, include_history=True
    )

    policy_mode = POLICY_MODE
    policy_result = None
    policy_source = "contextual_bandit"

    if policy_mode in {"ppo_first", "ppo_only"}:
        policy_result = run_ppo_policy(decision_point, normalized_context, actions)
        if policy_result:
            policy_source = "ppo_policy"
        elif policy_mode == "ppo_only":
            # Never silently change the study treatment after startup. A hard
            # failure lets monitoring pause recruitment and investigate.
            raise RuntimeError(
                "POLICY_MODE=ppo_only could not produce an eligible PPO action"
            )

    if policy_mode in {"offline_first", "offline_only", "ppo_first"}:
        if policy_result is None:
            policy_result = _run_offline_policy(decision_point, normalized_context, actions)
            if policy_result:
                policy_source = "offline_policy"
            elif policy_mode == "offline_only":
                policy_result = ("no-op", 1.0, "offline_unavailable_noop", {"match_type": "none"})
                policy_source = "offline_policy_noop"

    if policy_result is None:
        policy_result = _run_bandit_policy(db, decision_point, context_key, actions)
        policy_source = "contextual_bandit"

    action, propensity, model_name, model_meta = policy_result

    selected_stat = _get_or_create_stat(db, decision_point, context_key, action)
    if not FREEZE_POLICY:
        selected_stat.impressions += 1
        selected_stat.reward_sum -= ACTION_COST.get(action, 0.0)
        selected_stat.updated_at = datetime.utcnow()

    log = DecisionLog(
        session_id=session_id,
        decision_point=decision_point,
        action=action,
        propensity=propensity,
        context_json={
            "context_raw": context,
            "context_normalized": normalized_context,
            "context_key": context_key,
            "eligible_actions": actions,
            "model": model_name,
            "schema_version": CONTEXT_SCHEMA_VERSION,
            "policy_source": policy_source,
            "timing_gate": {
                "should_show": True,
                "reason": gate_reason,
                "next_check_after_ms": 0,
            },
            "opportunity": {
                "opportunity_id": context.get("opportunity_id"),
                "opportunity_type": context.get("opportunity_type"),
                "opportunity_index": context.get("opportunity_index"),
                "elapsed_ms": context.get("elapsed_ms"),
            },
            "frozen": FREEZE_POLICY,
            "worker_id": worker_id,
            "persona": persona,
            **model_meta,
        },
        timestamp=datetime.utcnow(),
    )
    db.add(log)
    db.commit()

    return {
        "decision_id": log.id,
        "action": action,
        "propensity": propensity,
        "decision_point": decision_point,
        "model": model_name,
        "should_show": action != "no-op",
        "next_check_after_ms": 0,
        "reason_code": "served",
    }


def apply_reward_from_event(
    db: Session,
    session_id: str,
    event_type: str,
    page: str,
    metadata: dict | None = None,
) -> None:
    """Credit a reward signal back to the most relevant recent bandit decision.

    Resolution priority:
    1. If ``metadata`` contains a ``decision_id``, the corresponding
       :class:`~app.models.DecisionLog` is updated directly.
    2. Otherwise, the most recent ``DecisionLog`` within the lookback window
       that matches the inferred or explicit decision point (and optional
       action) is updated.

    If ``reward`` is zero (unrecognised event) or no matching decision is
    found, the function returns without making any changes.

    Args:
        db: Active SQLAlchemy database session.
        session_id: Anonymous visitor session identifier.
        event_type: The type of user event that generated the reward.
        page: URL path where the event occurred (used to infer decision point).
        metadata: Optional dict that may contain ``decision_id``,
            ``decision_point``, ``action``, and event-specific data.
    """
    reward = _event_reward(event_type, metadata)
    if reward == 0 or FREEZE_POLICY:
        return

    md = metadata or {}
    now = datetime.utcnow()

    explicit_decision_id = md.get("decision_id")
    explicit_point = md.get("decision_point")
    explicit_action = md.get("action")
    inferred_point = _decision_point_from_page(page)

    if explicit_decision_id:
        log = (
            db.query(DecisionLog)
            .filter(
                DecisionLog.id == explicit_decision_id,
                DecisionLog.session_id == session_id,
            )
            .first()
        )
        if log:
            context_key = str((log.context_json or {}).get("context_key") or "")
            if context_key:
                stat = _get_or_create_stat(db, log.decision_point, context_key, log.action)
                stat.reward_sum += reward
                stat.updated_at = now
                db.commit()
        return

    q = (
        db.query(DecisionLog)
        .filter(
            DecisionLog.session_id == session_id,
            DecisionLog.timestamp >= now - timedelta(minutes=LOOKBACK_MINUTES),
        )
        .order_by(DecisionLog.timestamp.desc())
    )

    if explicit_point:
        q = q.filter(DecisionLog.decision_point == explicit_point)
    elif inferred_point:
        q = q.filter(DecisionLog.decision_point == inferred_point)

    if explicit_action:
        q = q.filter(DecisionLog.action == explicit_action)

    log = q.first()
    if not log:
        return

    context_key = str((log.context_json or {}).get("context_key") or "")
    if not context_key:
        return

    stat = _get_or_create_stat(db, log.decision_point, context_key, log.action)
    stat.reward_sum += reward
    stat.updated_at = now
    db.commit()
