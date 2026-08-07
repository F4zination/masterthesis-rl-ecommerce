import random
from datetime import datetime
from sqlalchemy.orm import Session
from ..models import DecisionLog

# Phase 1 action space — matches BanditDeploymentConcept.md
ACTIONS = ["no-op", "trending_carousel", "discount_banner", "frequently_bought_together"]


def get_decision(
    db: Session,
    session_id: str,
    decision_point: str,
    context: dict | None = None,
) -> dict:
    """Phase 1 stub: returns no-op with propensity 1.0.

    In Phase 2 this will be replaced by a contextual bandit.
    """
    action = "no-op"
    propensity = 1.0

    log = DecisionLog(
        session_id=session_id,
        decision_point=decision_point,
        action=action,
        propensity=propensity,
        context_json=context or {},
        timestamp=datetime.utcnow(),
    )
    db.add(log)
    db.commit()

    return {
        "action": action,
        "propensity": propensity,
        "decision_point": decision_point,
    }
