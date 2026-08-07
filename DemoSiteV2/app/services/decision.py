import random
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from ..models import BanditArmStat, DecisionLog
from ..config import FREEZE_POLICY
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
)


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

    # Frozen study serving must keep the policy tables read-only. A missing row still has the
    # documented prior mean through _estimate_mean(None); inserting an empty
    # row would mutate the release-locked bandit parameters even though it does
    # not change the selected action.
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


def get_decision(
    db: Session,
    session_id: str,
    decision_point: str,
    context: dict | None = None,
    worker_id: str = "",
    persona: str = "",
) -> dict:
    """Run the epsilon-greedy contextual bandit and return the selected action.

    Fetches arm statistics for all eligible actions (initialising missing rows
    only outside frozen mode), applies epsilon-greedy exploration, increments
    the impression counter, subtracts the action cost as an implicit penalty,
    and persists a :class:`~app.models.DecisionLog` entry.

    Args:
        db: Active SQLAlchemy database session.
        session_id: Anonymous visitor session identifier.
        decision_point: The step in the user journey (e.g., ``"pdp"``).
        context: Optional raw context dictionary supplied by the client.
        worker_id: Clickworker study participant id, stamped into the decision
            log's ``context_json`` for study attribution (not used by the
            policy / context key).
        persona: Clickworker study persona/scenario assignment, stamped into
            the decision log's ``context_json``.

    Returns:
        A dict with keys ``decision_id``, ``action``, ``propensity``,
        ``decision_point``, and ``model``.
    """
    if decision_point not in DECISION_POINTS:
        raise ValueError(f"unsupported decision_point: {decision_point!r}")

    actions = _eligible_actions(decision_point)
    context_key, normalized_context = _normalize_context(decision_point, context)

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
        propensity = 1.0
    else:
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

    if not FREEZE_POLICY:
        selected_stat = stats_by_action[action]
        selected_stat.impressions += 1
        selected_stat.reward_sum -= ACTION_COST.get(action, 0.0)
        selected_stat.updated_at = datetime.utcnow()

    log = DecisionLog(
        session_id=session_id,
        decision_point=decision_point,
        action=action,
        propensity=propensity,
        context_json={
            "context_raw": context or {},
            "context_normalized": normalized_context,
            "context_key": context_key,
            "eligible_actions": actions,
            "model": "epsilon_greedy_v2_frozen_greedy" if FREEZE_POLICY else "epsilon_greedy_v2",
            "schema_version": CONTEXT_SCHEMA_VERSION,
            "epsilon": 0.0 if FREEZE_POLICY else EPSILON,
            "estimates": estimates,
            "frozen": FREEZE_POLICY,
            "worker_id": worker_id,
            "persona": persona,
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
        "model": "epsilon_greedy_v2",
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
    if reward == 0:
        return

    md = metadata or {}
    now = datetime.utcnow()

    explicit_decision_id = md.get("decision_id")
    explicit_point = md.get("decision_point")
    explicit_action = md.get("action")
    inferred_point = _decision_point_from_page(page)

    if FREEZE_POLICY:
        return

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
