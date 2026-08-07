from datetime import datetime
from sqlalchemy.orm import Session
from ..models import Event


def log_event(
    db: Session,
    session_id: str,
    event_type: str,
    page: str = "",
    element: str = "",
    metadata: dict | None = None,
):
    event = Event(
        session_id=session_id,
        event_type=event_type,
        page=page,
        element=element,
        timestamp=datetime.utcnow(),
        metadata_json=metadata or {},
    )
    db.add(event)
    db.commit()
    return event
