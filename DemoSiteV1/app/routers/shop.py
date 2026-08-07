import random
from pathlib import Path
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Category, Product, Order, OrderItem
from ..services.cart import (
    get_cart_items, get_cart_total, get_cart_count, clear_cart,
)

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


def _session_id(request: Request) -> str:
    return request.cookies.get("session_id", "unknown")


def _base_context(request: Request, db: Session) -> dict:
    categories = db.query(Category).order_by(Category.name).all()
    cart_count = get_cart_count(db, _session_id(request))
    return {
        "categories": categories,
        "cart_count": cart_count,
        "session_id": _session_id(request),
    }


@router.get("/", response_class=HTMLResponse)
def home(request: Request, db: Session = Depends(get_db)):
    ctx = _base_context(request, db)
    products = db.query(Product).all()
    ctx["featured"] = random.sample(products, min(8, len(products)))
    return templates.TemplateResponse(request, name="home.html", context=ctx)


@router.get("/category/{slug}", response_class=HTMLResponse)
def category_page(slug: str, request: Request, db: Session = Depends(get_db)):
    ctx = _base_context(request, db)
    category = db.query(Category).filter(Category.slug == slug).first()
    if not category:
        return HTMLResponse("Category not found", status_code=404)
    products = db.query(Product).filter(Product.category_id == category.id).all()
    ctx["category"] = category
    ctx["products"] = products
    return templates.TemplateResponse(request, name="category.html", context=ctx)


@router.get("/product/{slug}", response_class=HTMLResponse)
def product_page(slug: str, request: Request, db: Session = Depends(get_db)):
    ctx = _base_context(request, db)
    product = db.query(Product).filter(Product.slug == slug).first()
    if not product:
        return HTMLResponse("Product not found", status_code=404)
    related = (
        db.query(Product)
        .filter(Product.category_id == product.category_id, Product.id != product.id)
        .limit(4)
        .all()
    )
    ctx["product"] = product
    ctx["related"] = related
    return templates.TemplateResponse(request, name="product.html", context=ctx)


@router.get("/cart", response_class=HTMLResponse)
def cart_page(request: Request, db: Session = Depends(get_db)):
    ctx = _base_context(request, db)
    items = get_cart_items(db, _session_id(request))
    ctx["items"] = items
    ctx["total"] = get_cart_total(db, _session_id(request))
    return templates.TemplateResponse(request, name="cart.html", context=ctx)


@router.get("/checkout", response_class=HTMLResponse)
def checkout_page(request: Request, db: Session = Depends(get_db)):
    ctx = _base_context(request, db)
    items = get_cart_items(db, _session_id(request))
    if not items:
        return RedirectResponse("/cart", status_code=303)
    ctx["items"] = items
    ctx["total"] = get_cart_total(db, _session_id(request))
    return templates.TemplateResponse(request, name="checkout.html", context=ctx)


@router.post("/checkout")
def process_checkout(request: Request, db: Session = Depends(get_db)):
    sid = _session_id(request)
    items = get_cart_items(db, sid)
    if not items:
        return RedirectResponse("/cart", status_code=303)

    total = get_cart_total(db, sid)
    order = Order(session_id=sid, total=total)
    db.add(order)
    db.flush()

    for cart_item in items:
        oi = OrderItem(
            order_id=order.id,
            product_id=cart_item.product_id,
            quantity=cart_item.quantity,
            price=cart_item.product.price,
        )
        db.add(oi)

    db.commit()
    clear_cart(db, sid)

    return RedirectResponse(f"/confirmation/{order.id}", status_code=303)


@router.get("/confirmation/{order_id}", response_class=HTMLResponse)
def confirmation_page(order_id: int, request: Request, db: Session = Depends(get_db)):
    ctx = _base_context(request, db)
    order = db.query(Order).filter(Order.id == order_id).first()
    if not order:
        return HTMLResponse("Order not found", status_code=404)
    ctx["order"] = order
    return templates.TemplateResponse(request, name="confirmation.html", context=ctx)
