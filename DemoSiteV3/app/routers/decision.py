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
    opportunity_id: str | None = None
    opportunity_type: str | None = None
    opportunity_index: int | None = None


@router.post("")
def request_decision(body: DecisionRequest, request: Request, db: Session = Depends(get_db)):
    """Request a bandit decision for a visitor at a specific funnel step.

    Delegates to :func:`~app.services.decision.get_decision`, which runs the
    epsilon-greedy policy and returns the chosen widget action together with
    its propensity score.

    Args:
        body: Validated :class:`DecisionRequest` payload.
        db: Injected database session.

    Returns:
        A dict with ``decision_id``, ``action``, ``propensity``,
        ``decision_point``, and ``model``.
    """
    context = dict(body.context or {})
    if body.opportunity_id is not None:
        context.setdefault("opportunity_id", body.opportunity_id)
    if body.opportunity_type is not None:
        context.setdefault("opportunity_type", body.opportunity_type)
    if body.opportunity_index is not None:
        context.setdefault("opportunity_index", body.opportunity_index)

    result = get_decision(
        db,
        session_id=body.session_id,
        decision_point=body.decision_point,
        context=context,
        worker_id=request.cookies.get("wid", ""),
        persona=request.cookies.get("study_persona", ""),
    )
    return result
