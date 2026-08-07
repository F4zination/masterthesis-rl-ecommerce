from sqlalchemy.orm import Session
from ..models import CartItem, Product


def get_cart_items(db: Session, session_id: str):
    return (
        db.query(CartItem)
        .filter(CartItem.session_id == session_id)
        .all()
    )


def add_to_cart(db: Session, session_id: str, product_id: int, quantity: int = 1):
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
    item = (
        db.query(CartItem)
        .filter(CartItem.session_id == session_id, CartItem.product_id == product_id)
        .first()
    )
    if item:
        db.delete(item)
        db.commit()


def clear_cart(db: Session, session_id: str):
    db.query(CartItem).filter(CartItem.session_id == session_id).delete()
    db.commit()


def get_cart_total(db: Session, session_id: str) -> float:
    items = get_cart_items(db, session_id)
    return sum(item.product.price * item.quantity for item in items)


def get_cart_count(db: Session, session_id: str) -> int:
    items = get_cart_items(db, session_id)
    return sum(item.quantity for item in items)
