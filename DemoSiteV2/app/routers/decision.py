from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from shared_schema.constants import DecisionPointName
from sqlalchemy.orm import Session

from ..database import get_db
from ..services.decision import get_decision

router = APIRouter(prefix="/api/decision", tags=["decision"])


class DecisionRequest(BaseModel):
    """Request body for the bandit decision endpoint.

    Attributes:
        session_id: Anonymous visitor session identifier.
        decision_point: The location in the user journey requesting a decision
            (e.g., ``"landing"``, ``"pdp"``, ``"cart"``).
        context: Optional dict of client-side contextual features such as
            ``screen_width``, ``referrer``, ``cart_total``, and ``page_depth``.
    """

    session_id: str
    decision_point: DecisionPointName
    context: dict | None = None


@router.post("")
def request_decision(body: DecisionRequest, request: Request, db: Session = Depends(get_db)):
    """Request a bandit decision for a visitor at a specific funnel step.

    Delegates to :func:`~app.services.decision.get_decision`, which runs the
    epsilon-greedy policy and returns the chosen widget action together with
    its propensity score. The Clickworker ``wid``/``study_persona`` cookies
    are read server-side and stamped into the decision log for attribution.

    Args:
        body: Validated :class:`DecisionRequest` payload.
        request: Incoming request (used to read study attribution cookies).
        db: Injected database session.

    Returns:
        A dict with ``decision_id``, ``action``, ``propensity``,
        ``decision_point``, and ``model``.
    """
    result = get_decision(
        db,
        session_id=body.session_id,
        decision_point=body.decision_point,
        context=body.context,
        worker_id=request.cookies.get("wid", ""),
        persona=request.cookies.get("study_persona", ""),
    )
    return result
