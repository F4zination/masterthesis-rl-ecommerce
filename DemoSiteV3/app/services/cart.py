from sqlalchemy.orm import Session
from ..models import CartItem, Product, SessionDiscount


def get_cart_items(db: Session, session_id: str):
    """Retrieve all cart items for the given session.

    Args:
        db: Active SQLAlchemy database session.
        session_id: Anonymous visitor session identifier.

    Returns:
        List of :class:`~app.models.CartItem` ORM instances with their
        related ``Product`` accessible via the ORM relationship.
    """
    return (
        db.query(CartItem)
        .filter(CartItem.session_id == session_id)
        .all()
    )


def add_to_cart(db: Session, session_id: str, product_id: int, quantity: int = 1):
    """Add a product to the session's cart or increment its quantity.

    If the product is already in the cart, its quantity is incremented by
    ``quantity``.  Otherwise a new :class:`~app.models.CartItem` row is
    created and committed.

    Args:
        db: Active SQLAlchemy database session.
        session_id: Anonymous visitor session identifier.
        product_id: Primary key of the product to add.
        quantity: Number of units to add (defaults to 1).
    """
    existing = (
        db.query(CartItem)
        .filter(CartItem.session_id == session_id, CartItem.product_id == product_id)
        .first()
    )
    if existing:
        existing.quantity += quantity
    else:
        item = CartItem(session_id=session_id, product_id=product_id, quantity=quantity)
        db.add(item)
    db.commit()


def update_cart_item(db: Session, session_id: str, product_id: int, quantity: int):
    """Set the quantity of an existing cart item, or remove it if zero.

    If ``quantity`` is less than or equal to zero the item is deleted from
    the cart.  Has no effect if the item does not exist.

    Args:
        db: Active SQLAlchemy database session.
        session_id: Anonymous visitor session identifier.
        product_id: Primary key of the product to update.
        quantity: New desired quantity; ≤ 0 removes the item.
    """
    item = (
        db.query(CartItem)
        .filter(CartItem.session_id == session_id, CartItem.product_id == product_id)
        .first()
    )
    if item:
        if quantity <= 0:
            db.delete(item)
        else:
            item.quantity = quantity
        db.commit()


def remove_from_cart(db: Session, session_id: str, product_id: int):
    """Remove a product from the session's cart entirely.

    Has no effect if the product is not currently in the cart.

    Args:
        db: Active SQLAlchemy database session.
        session_id: Anonymous visitor session identifier.
        product_id: Primary key of the product to remove.
    """
    item = (
        db.query(CartItem)
        .filter(CartItem.session_id == session_id, CartItem.product_id == product_id)
        .first()
    )
    if item:
        db.delete(item)
        db.commit()


def clear_cart(db: Session, session_id: str):
    """Delete all cart items belonging to the given session.

    Called after a successful checkout to empty the cart.

    Args:
        db: Active SQLAlchemy database session.
        session_id: Anonymous visitor session identifier.
    """
    db.query(CartItem).filter(CartItem.session_id == session_id).delete()
    db.commit()


def get_cart_total(db: Session, session_id: str) -> float:
    """Calculate the pre-discount subtotal for the session's cart.

    Args:
        db: Active SQLAlchemy database session.
        session_id: Anonymous visitor session identifier.

    Returns:
        Sum of ``price * quantity`` for all items in the cart.
    """
    items = get_cart_items(db, session_id)
    return sum(item.product.price * item.quantity for item in items)


def get_cart_count(db: Session, session_id: str) -> int:
    """Return the total number of product units in the session's cart.

    Args:
        db: Active SQLAlchemy database session.
        session_id: Anonymous visitor session identifier.

    Returns:
        Sum of all item quantities (not the number of distinct products).
    """
    items = get_cart_items(db, session_id)
    return sum(item.quantity for item in items)


# ── Discount helpers ──────────────────────────────────────────────────────────

def get_session_discount(db: Session, session_id: str) -> float:
    """Return the active discount percentage for the session, or 0.0.

    Args:
        db: Active SQLAlchemy database session.
        session_id: Anonymous visitor session identifier.

    Returns:
        The discount percentage (e.g., ``10.0`` for 10 %), or ``0.0`` if no
        discount has been applied to this session.
    """
    d = db.query(SessionDiscount).filter(SessionDiscount.session_id == session_id).first()
    return d.discount_pct if d else 0.0


def apply_session_discount(db: Session, session_id: str, discount_pct: float, code: str = "") -> None:
    """Attach or update a discount for the given session.

    If the session already has a discount record it is updated in-place;
    otherwise a new :class:`~app.models.SessionDiscount` row is created.

    Args:
        db: Active SQLAlchemy database session.
        session_id: Anonymous visitor session identifier.
        discount_pct: Percentage to discount (e.g., ``10.0`` for 10 %).
        code: Optional promotional code string to associate with the discount.
    """
    existing = db.query(SessionDiscount).filter(SessionDiscount.session_id == session_id).first()
    if existing:
        existing.discount_pct = discount_pct
        existing.code = code
    else:
        db.add(SessionDiscount(session_id=session_id, discount_pct=discount_pct, code=code))
    db.commit()


def clear_discount(db: Session, session_id: str) -> None:
    """Remove any active discount for the given session.

    Called after a successful checkout so that the discount cannot be reused
    in a future session.

    Args:
        db: Active SQLAlchemy database session.
        session_id: Anonymous visitor session identifier.
    """
    db.query(SessionDiscount).filter(SessionDiscount.session_id == session_id).delete()
    db.commit()
