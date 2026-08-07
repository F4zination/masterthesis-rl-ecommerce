from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ..services.tracking import log_event

router = APIRouter(prefix="/api/events", tags=["events"])


class EventRequest(BaseModel):
    session_id: str
    event_type: str
    page: str = ""
    element: str = ""
    metadata: dict | None = None


@router.post("")
def track_event(body: EventRequest, db: Session = Depends(get_db)):
    event = log_event(
        db,
        session_id=body.session_id,
        event_type=body.event_type,
        page=body.page,
        element=body.element,
        metadata=body.metadata,
    )
    return {"status": "ok", "event_id": event.id}
