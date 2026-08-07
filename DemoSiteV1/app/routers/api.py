from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ..services.cart import (
    get_cart_items, add_to_cart, update_cart_item, remove_from_cart,
    get_cart_total, get_cart_count,
)

router = APIRouter(prefix="/api/cart", tags=["cart"])


class CartAddRequest(BaseModel):
    product_id: int
    quantity: int = 1


class CartUpdateRequest(BaseModel):
    product_id: int
    quantity: int


class CartRemoveRequest(BaseModel):
    product_id: int


def _session_id(request: Request) -> str:
    return request.cookies.get("session_id", "unknown")


@router.get("")
def get_cart(request: Request, db: Session = Depends(get_db)):
    sid = _session_id(request)
    items = get_cart_items(db, sid)
    return {
        "items": [
            {
                "product_id": i.product_id,
                "name": i.product.name,
                "price": i.product.price,
                "quantity": i.quantity,
                "subtotal": round(i.product.price * i.quantity, 2),
                "image_url": i.product.image_url,
                "slug": i.product.slug,
            }
            for i in items
        ],
        "total": round(get_cart_total(db, sid), 2),
        "count": get_cart_count(db, sid),
    }


@router.post("/add")
def cart_add(body: CartAddRequest, request: Request, db: Session = Depends(get_db)):
    sid = _session_id(request)
    add_to_cart(db, sid, body.product_id, body.quantity)
    return {"status": "ok", "count": get_cart_count(db, sid)}


@router.post("/update")
def cart_update(body: CartUpdateRequest, request: Request, db: Session = Depends(get_db)):
    sid = _session_id(request)
    update_cart_item(db, sid, body.product_id, body.quantity)
    return {"status": "ok", "count": get_cart_count(db, sid)}


@router.post("/remove")
def cart_remove(body: CartRemoveRequest, request: Request, db: Session = Depends(get_db)):
    sid = _session_id(request)
    remove_from_cart(db, sid, body.product_id)
    return {"status": "ok", "count": get_cart_count(db, sid)}
