from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ..services.tracking import log_event

router = APIRouter(prefix="/api/events", tags=["events"])


class EventRequest(BaseModel):
    """Request body for the event tracking endpoint.

    Attributes:
        session_id: Anonymous visitor session identifier.
        event_type: Semantic label for the interaction (e.g., ``"add_to_cart"``).
        page: URL path where the event occurred.
        element: Optional identifier of the interacted DOM element.
        metadata: Optional dict of additional event-specific data (e.g.,
            ``decision_id``, ``depth``, ``order_total``).
    """

    session_id: str
    event_type: str
    page: str = ""
    element: str = ""
    metadata: dict | None = None


@router.post("")
def track_event(body: EventRequest, request: Request, db: Session = Depends(get_db)):
    """Record a user interaction event and propagate its reward signal.

    Persists the event via :func:`~app.services.tracking.log_event`, which
    also calls the bandit reward attribution logic in the same request. The
    Clickworker ``wid``/``study_persona`` cookies (set on the landing page)
    are read server-side and stamped onto every event so collected sessions
    can be joined back to their experiment cell and persona.

    Args:
        body: Validated :class:`EventRequest` payload.
        request: Incoming request (used to read study attribution cookies).
        db: Injected database session.

    Returns:
        A dict with ``status: "ok"`` and the new event's ``event_id``.
    """
    event = log_event(
        db,
        session_id=body.session_id,
        event_type=body.event_type,
        page=body.page,
        element=body.element,
        metadata=body.metadata,
        worker_id=request.cookies.get("wid", ""),
        persona=request.cookies.get("study_persona", ""),
    )
    return {"status": "ok", "event_id": event.id}
