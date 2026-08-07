from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ..services.decision import get_decision

router = APIRouter(prefix="/api/decision", tags=["decision"])


class DecisionRequest(BaseModel):
    session_id: str
    decision_point: str
    context: dict | None = None


@router.post("")
def request_decision(body: DecisionRequest, db: Session = Depends(get_db)):
    result = get_decision(
        db,
        session_id=body.session_id,
        decision_point=body.decision_point,
        context=body.context,
    )
    return result
