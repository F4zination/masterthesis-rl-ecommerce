from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ..services.images import thumb_url
from ..services.cart import (
    get_cart_items, add_to_cart, update_cart_item, remove_from_cart,
    get_cart_total, get_cart_count, get_session_discount, apply_session_discount,
)

router = APIRouter(prefix="/api/cart", tags=["cart"])

# The single promo code the shop honours. The discount_banner widget advertises
# it; the promo box on cart/checkout redeems it.
DISCOUNT_CODE = "SAVE10"


class CartAddRequest(BaseModel):
    """Request body for adding a product to the cart.

    Attributes:
        product_id: Primary key of the product to add.
        quantity: Number of units to add (defaults to 1).
    """

    product_id: int
    quantity: int = 1


class CartUpdateRequest(BaseModel):
    """Request body for updating a cart item's quantity.

    Attributes:
        product_id: Primary key of the product to update.
        quantity: New desired quantity; setting this to ≤ 0 removes the item.
    """

    product_id: int
    quantity: int


class DiscountRequest(BaseModel):
    """Request body for redeeming a promo code.

    Attributes:
        code: Promo code typed by the shopper. Omitted by the discount widget,
            which redeems the code directly.
    """

    code: str | None = None


class CartRemoveRequest(BaseModel):
    """Request body for removing a product from the cart.

    Attributes:
        product_id: Primary key of the product to remove.
    """

    product_id: int


def _session_id(request: Request) -> str:
    """Extract the session identifier from the request cookie.

    Args:
        request: The incoming FastAPI ``Request`` object.

    Returns:
        The ``session_id`` cookie value, or ``"unknown"`` if absent.
    """
    return request.cookies.get("session_id", "unknown")


@router.get("")
def get_cart(request: Request, db: Session = Depends(get_db)):
    """Return the current cart contents with pricing and discount details.

    Args:
        request: Incoming request (used to read the session cookie).
        db: Injected database session.

    Returns:
        A dict containing ``items``, ``subtotal``, ``discount_pct``,
        ``discount_amount``, ``total``, and ``count``.
    """
    sid = _session_id(request)
    items = get_cart_items(db, sid)
    subtotal = round(get_cart_total(db, sid), 2)
    discount_pct = get_session_discount(db, sid)
    discount_amount = round(subtotal * discount_pct / 100, 2)
    return {
        "items": [
            {
                "product_id": i.product_id,
                "name": i.product.name,
                "price": i.product.price,
                "quantity": i.quantity,
                "subtotal": round(i.product.price * i.quantity, 2),
                "image_url": i.product.image_url,
                "thumb_url": thumb_url(i.product.image_url),
                "slug": i.product.slug,
            }
            for i in items
        ],
        "subtotal": subtotal,
        "discount_pct": discount_pct,
        "discount_amount": discount_amount,
        "total": round(subtotal - discount_amount, 2),
        "count": get_cart_count(db, sid),
    }


@router.post("/add")
def cart_add(body: CartAddRequest, request: Request, db: Session = Depends(get_db)):
    """Add a product to the session's cart.

    Args:
        body: Validated :class:`CartAddRequest` payload.
        request: Incoming request (used to read the session cookie).
        db: Injected database session.

    Returns:
        A dict with ``status: "ok"`` and the updated cart ``count``.
    """
    sid = _session_id(request)
    add_to_cart(db, sid, body.product_id, body.quantity)
    return {"status": "ok", "count": get_cart_count(db, sid)}


@router.post("/update")
def cart_update(body: CartUpdateRequest, request: Request, db: Session = Depends(get_db)):
    """Update the quantity of a product already in the cart.

    Args:
        body: Validated :class:`CartUpdateRequest` payload.
        request: Incoming request (used to read the session cookie).
        db: Injected database session.

    Returns:
        A dict with ``status: "ok"`` and the updated cart ``count``.
    """
    sid = _session_id(request)
    update_cart_item(db, sid, body.product_id, body.quantity)
    return {"status": "ok", "count": get_cart_count(db, sid)}


@router.post("/remove")
def cart_remove(body: CartRemoveRequest, request: Request, db: Session = Depends(get_db)):
    """Remove a product from the session's cart.

    Args:
        body: Validated :class:`CartRemoveRequest` payload.
        request: Incoming request (used to read the session cookie).
        db: Injected database session.

    Returns:
        A dict with ``status: "ok"`` and the updated cart ``count``.
    """
    sid = _session_id(request)
    remove_from_cart(db, sid, body.product_id)
    return {"status": "ok", "count": get_cart_count(db, sid)}


@router.get("/discount")
def get_discount(request: Request, db: Session = Depends(get_db)):
    """Return the current discount status for the session's cart.

    Args:
        request: Incoming request (used to read the session cookie).
        db: Injected database session.

    Returns:
        A dict with ``discount_pct``, ``discount_amount``, ``total``,
        and ``already_applied`` (bool).
    """
    sid = _session_id(request)
    subtotal = round(get_cart_total(db, sid), 2)
    discount_pct = get_session_discount(db, sid)
    discount_amount = round(subtotal * discount_pct / 100, 2)
    return {
        "discount_pct": discount_pct,
        "discount_amount": discount_amount,
        "total": round(subtotal - discount_amount, 2),
        "already_applied": discount_pct > 0,
    }


@router.post("/discount")
def apply_discount(
    request: Request,
    body: DiscountRequest | None = None,
    db: Session = Depends(get_db),
):
    """Apply the 10% ``SAVE10`` discount to the session's cart (once per session).

    Two callers share this endpoint:

    * The ``discount_banner`` widget posts **no** body — clicking "Apply
      Discount" redeems the code directly, exactly as before.
    * The promo-code box on the cart/checkout page posts a ``code``, which must
      match ``SAVE10`` to be accepted.

    The manual box only redeems a code the participant has already been shown by
    the widget, so it does not hand the discount to sessions that never saw that
    arm — it answers "where do I enter my code?" without changing what the
    discount arm is worth. Manual redemption deliberately emits no widget reward.

    Args:
        request: Incoming request (used to read the session cookie).
        body: Optional payload carrying a manually entered promo ``code``.
        db: Injected database session.

    Returns:
        A dict describing the outcome with keys ``status``, ``discount_pct``,
        ``discount_amount``, ``new_total``, and ``message``.
    """
    sid = _session_id(request)
    submitted = (body.code if body and body.code else "").strip().upper()
    if submitted and submitted != DISCOUNT_CODE:
        return {
            "status": "invalid_code",
            "message": f"“{submitted}” is not a valid promo code.",
        }
    if get_session_discount(db, sid) > 0:
        return {"status": "already_applied", "message": "Discount already applied to your cart."}
    apply_session_discount(db, sid, 10.0, DISCOUNT_CODE)
    subtotal = round(get_cart_total(db, sid), 2)
    discount_amount = round(subtotal * 0.10, 2)
    return {
        "status": "ok",
        "discount_pct": 10.0,
        "discount_amount": discount_amount,
        "new_total": round(subtotal - discount_amount, 2),
        "message": f"10% discount applied! Code: {DISCOUNT_CODE}",
    }
