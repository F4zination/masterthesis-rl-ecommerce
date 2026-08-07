from datetime import datetime
from sqlalchemy.orm import Session
from ..models import Event
from .decision import apply_reward_from_event


def log_event(
    db: Session,
    session_id: str,
    event_type: str,
    page: str = "",
    element: str = "",
    metadata: dict | None = None,
    worker_id: str = "",
    persona: str = "",
):
    """Persist a user interaction event and propagate the reward signal.

    Creates an :class:`~app.models.Event` record in the database and
    immediately calls :func:`~app.services.decision.apply_reward_from_event`
    so bandit arm statistics are updated within the same request cycle.

    Args:
        db: Active SQLAlchemy database session.
        session_id: Anonymous visitor session identifier.
        event_type: Semantic label for the interaction (e.g., ``"add_to_cart"``).
        page: URL path where the event occurred.
        element: Optional identifier of the interacted DOM element.
        metadata: Optional dict of additional event-specific data.
        worker_id: Clickworker study participant id (study attribution).
        persona: Clickworker study persona/scenario assignment.

    Returns:
        The persisted :class:`~app.models.Event` instance.
    """
    event = Event(
        session_id=session_id,
        event_type=event_type,
        page=page,
        element=element,
        timestamp=datetime.utcnow(),
        metadata_json=metadata or {},
        worker_id=worker_id,
        persona=persona,
    )
    db.add(event)
    db.commit()

    apply_reward_from_event(
        db,
        session_id=session_id,
        event_type=event_type,
        page=page,
        metadata=metadata,
    )
    return event
