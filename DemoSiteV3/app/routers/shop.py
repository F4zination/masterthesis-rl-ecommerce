import hashlib
import random
import uuid
from collections import Counter
from pathlib import Path
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..config import LEGAL_BASE_URL
from ..database import get_db
from ..models import Category, Product, Order, OrderItem
from ..services.cart import (
    get_cart_items, get_cart_total, get_cart_count, clear_cart,
    get_session_discount, clear_discount,
)
from ..services.tracking import log_event
from ..services.images import register_filters, thumb_url

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))
register_filters(templates)


def _session_id(request: Request) -> str:
    """Extract the session identifier from the request cookie.

    Args:
        request: The incoming FastAPI ``Request`` object.

    Returns:
        The ``session_id`` cookie value, or ``"unknown"`` if absent.
    """
    return request.cookies.get("session_id", "unknown")


def _study_attribution(request: Request) -> tuple[str, str]:
    """Read the Clickworker study ``wid``/``persona`` cookies.

    These cookies are set on the landing page from the worker's unique URL
    and are used to attribute every session to its experiment cell.

    Args:
        request: The incoming FastAPI ``Request`` object.

    Returns:
        A ``(worker_id, persona)`` tuple; either value is ``""`` if absent.
    """
    return request.cookies.get("wid", ""), request.cookies.get("study_persona", "")


def _base_context(request: Request, db: Session) -> dict:
    """Build the template context variables shared by all shop pages.

    Fetches the full category list and the current cart item count so that
    the navigation bar is always up-to-date.

    Args:
        request: The incoming FastAPI ``Request`` object.
        db: Active SQLAlchemy database session.

    Returns:
        A dict containing ``categories``, ``cart_count``, and ``session_id``.
    """
    categories = db.query(Category).order_by(Category.name).all()
    cart_count = get_cart_count(db, _session_id(request))
    worker_id, persona = _study_attribution(request)
    return {
        "categories": categories,
        "cart_count": cart_count,
        "session_id": _session_id(request),
        "study_persona": persona,
        "study_wid": worker_id,
        "legal_base_url": LEGAL_BASE_URL,
    }


@router.get("/api/trending")
def api_trending(db: Session = Depends(get_db)):
    """Return a random sample of up to six products for the trending widget.

    Args:
        db: Injected database session.

    Returns:
        A list of dicts each containing ``id``, ``name``, ``slug``,
        ``price``, and ``image_url``.
    """
    products = db.query(Product).all()
    sample = random.sample(products, min(6, len(products)))
    return [
        {"id": p.id, "name": p.name, "slug": p.slug,
         "price": p.price, "image_url": p.image_url,
         "thumb_url": thumb_url(p.image_url)}
        for p in sample
    ]


_FBT_LIMIT = 6


@router.get("/api/frequently_bought_together")
def api_frequently_bought_together(
    request: Request,
    product_id: int | None = None,
    db: Session = Depends(get_db),
):
    """Return products that genuinely go with what the shopper is looking at.

    Recommendations are anchored on the product being viewed (``product_id``)
    or, where no anchor is supplied (cart/checkout), on the session's current
    cart contents. Ranking is by true co-occurrence — how often a product
    appeared in the *same order* as an anchor item.

    Order history is empty at the start of a study run, so the remaining slots
    are filled from the anchor's own category before falling back to overall
    best-sellers. Without that category step the widget showed an unrelated
    random assortment, which participants read as broken recommendations.

    Args:
        request: Incoming request (used to read the session cookie).
        product_id: Optional anchor product the shopper is currently viewing.
        db: Injected database session.

    Returns:
        A list of up to six product dicts each containing ``id``, ``name``,
        ``slug``, ``price``, ``image_url``, and ``thumb_url``.
    """
    if product_id is not None:
        anchor_ids = [product_id]
    else:
        anchor_ids = [i.product_id for i in get_cart_items(db, _session_id(request))]

    anchors = db.query(Product).filter(Product.id.in_(anchor_ids)).all() if anchor_ids else []
    anchor_set = {p.id for p in anchors}
    ranked: list[int] = []

    def _remaining() -> int:
        return _FBT_LIMIT - len(ranked)

    # 1. Co-occurrence: products that shared an order with an anchor item.
    if anchor_set:
        order_ids = [
            oid for (oid,) in db.query(OrderItem.order_id)
            .filter(OrderItem.product_id.in_(anchor_set))
            .distinct()
        ]
        if order_ids:
            counts: Counter[int] = Counter()
            for row in db.query(OrderItem).filter(OrderItem.order_id.in_(order_ids)).all():
                if row.product_id not in anchor_set:
                    counts[row.product_id] += row.quantity
            ranked = [pid for pid, _ in counts.most_common(_FBT_LIMIT)]

    # 2. Same-category fill — the sensible default while order history is thin.
    if _remaining() > 0 and anchors:
        exclude = anchor_set | set(ranked)
        siblings = (
            db.query(Product)
            .filter(
                Product.category_id.in_({p.category_id for p in anchors}),
                Product.id.notin_(exclude),
            )
            .all()
        )
        random.shuffle(siblings)
        ranked.extend(p.id for p in siblings[: _remaining()])

    # 3. Best-sellers, then random, so the widget always has something to show.
    if _remaining() > 0:
        exclude = anchor_set | set(ranked)
        popular: Counter[int] = Counter()
        for row in db.query(OrderItem).all():
            if row.product_id not in exclude:
                popular[row.product_id] += row.quantity
        ranked.extend(pid for pid, _ in popular.most_common(_remaining()))

    if _remaining() > 0:
        exclude = anchor_set | set(ranked)
        query = db.query(Product)
        if exclude:
            query = query.filter(Product.id.notin_(exclude))
        filler = query.all()
        if filler:
            ranked.extend(p.id for p in random.sample(filler, min(_remaining(), len(filler))))

    by_id = {p.id: p for p in db.query(Product).filter(Product.id.in_(ranked)).all()}
    return [
        {"id": p.id, "name": p.name, "slug": p.slug,
         "price": p.price, "image_url": p.image_url,
         "thumb_url": thumb_url(p.image_url)}
        for p in (by_id[pid] for pid in ranked if pid in by_id)
    ]


_VALID_PERSONAS = {"explorer", "fastbuyer", "detailedcomparator", "discounthunter", "windowshopper"}


@router.get("/", response_class=HTMLResponse)
def home(request: Request, db: Session = Depends(get_db),
         wid: str = "", persona: str = ""):
    ctx = _base_context(request, db)
    products = db.query(Product).all()
    ctx["featured"] = random.sample(products, min(8, len(products)))

    incoming_wid = wid.strip()
    existing_wid = request.cookies.get("wid", "")
    effective_wid = incoming_wid or existing_wid
    raw_persona = persona.strip().lower() or request.cookies.get("study_persona", "")
    effective_persona = raw_persona if raw_persona in _VALID_PERSONAS else ""
    # A different dispatcher worker must never inherit a long-lived browser
    # session from an earlier visit.  Preserve the session when the same worker
    # refreshes or resumes their assigned URL, but start a fresh one when the
    # incoming worker id differs from the study cookie.
    reset_study_session = bool(incoming_wid and incoming_wid != existing_wid)
    new_session_id = uuid.uuid4().hex if reset_study_session else ""
    if new_session_id:
        ctx["session_id"] = new_session_id
        # _base_context counted the *previous* session's cart. Rendering that
        # count next to a freshly reset session made a re-registered participant
        # believe their old items were still in the basket.
        ctx["cart_count"] = 0
    ctx["study_persona"] = effective_persona
    ctx["study_wid"] = effective_wid

    response = templates.TemplateResponse(request, name="home.html", context=ctx)
    if new_session_id:
        response.set_cookie(
            "session_id", new_session_id, max_age=3600, samesite="lax"
        )
    if effective_wid:
        response.set_cookie("wid", effective_wid, max_age=3600, httponly=True, samesite="lax")
    if effective_persona:
        response.set_cookie("study_persona", effective_persona, max_age=3600, httponly=True, samesite="lax")
    return response


@router.get("/category/{slug}", response_class=HTMLResponse)
def category_page(slug: str, request: Request, db: Session = Depends(get_db)):
    """Render the product listing page for a given category.

    Args:
        slug: URL-friendly category identifier.
        request: Incoming request (used for the template and session cookie).
        db: Injected database session.

    Returns:
        An HTML response rendering ``category.html``, or a 404 response if
        the category slug does not exist.
    """
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
    """Render the product detail page (PDP) for a given product.

    Also fetches up to four related products from the same category to
    display in the recommendations section.

    Args:
        slug: URL-friendly product identifier.
        request: Incoming request (used for the template and session cookie).
        db: Injected database session.

    Returns:
        An HTML response rendering ``product.html``, or a 404 response if
        the product slug does not exist.
    """
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
    """Render the shopping cart page with itemised pricing and discount info.

    Args:
        request: Incoming request (used for the template and session cookie).
        db: Injected database session.

    Returns:
        An HTML response rendering ``cart.html``.
    """
    ctx = _base_context(request, db)
    sid = _session_id(request)
    items = get_cart_items(db, sid)
    subtotal = get_cart_total(db, sid)
    discount_pct = get_session_discount(db, sid)
    discount_amount = round(subtotal * discount_pct / 100, 2)
    ctx["items"] = items
    ctx["subtotal"] = subtotal
    ctx["discount_pct"] = discount_pct
    ctx["discount_amount"] = discount_amount
    ctx["total"] = round(subtotal - discount_amount, 2)
    return templates.TemplateResponse(request, name="cart.html", context=ctx)


@router.get("/checkout", response_class=HTMLResponse)
def checkout_page(request: Request, db: Session = Depends(get_db)):
    """Render the checkout page, or redirect to the cart if it is empty.

    Args:
        request: Incoming request (used for the template and session cookie).
        db: Injected database session.

    Returns:
        An HTML response rendering ``checkout.html``, or a 303 redirect to
        ``/cart`` if the session's cart is empty.
    """
    ctx = _base_context(request, db)
    sid = _session_id(request)
    items = get_cart_items(db, sid)
    if not items:
        return RedirectResponse("/cart", status_code=303)
    subtotal = get_cart_total(db, sid)
    discount_pct = get_session_discount(db, sid)
    discount_amount = round(subtotal * discount_pct / 100, 2)
    ctx["items"] = items
    ctx["subtotal"] = subtotal
    ctx["discount_pct"] = discount_pct
    ctx["discount_amount"] = discount_amount
    ctx["total"] = round(subtotal - discount_amount, 2)
    ctx["item_count"] = sum(i.quantity for i in items)
    return templates.TemplateResponse(request, name="checkout.html", context=ctx)


@router.post("/checkout")
def process_checkout(request: Request, db: Session = Depends(get_db)):
    """Process the checkout form submission and create an order.

    Validates that the cart is non-empty, applies any active session discount,
    creates :class:`~app.models.Order` and :class:`~app.models.OrderItem`
    records, clears the cart and discount, emits a ``purchase`` event for
    bandit reward attribution, and redirects to the confirmation page.

    Args:
        request: Incoming request (used to read the session cookie).
        db: Injected database session.

    Returns:
        A 303 redirect to ``/confirmation/{order_id}``, or a redirect to
        ``/cart`` if the cart is empty.
    """
    sid = _session_id(request)
    worker_id, persona = _study_attribution(request)
    items = get_cart_items(db, sid)
    if not items:
        return RedirectResponse("/cart", status_code=303)

    total_raw = get_cart_total(db, sid)
    discount_pct = get_session_discount(db, sid)
    discount_amount = round(total_raw * discount_pct / 100, 2)
    total = round(total_raw - discount_amount, 2)
    order = Order(session_id=sid, total=total, worker_id=worker_id, persona=persona)
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
    clear_discount(db, sid)

    log_event(
        db,
        session_id=sid,
        event_type="purchase",
        page="/checkout",
        metadata={"order_id": order.id, "order_total": total},
        worker_id=worker_id,
        persona=persona,
    )

    return RedirectResponse(f"/confirmation/{order.id}", status_code=303)


@router.get("/confirmation/{order_id}", response_class=HTMLResponse)
def confirmation_page(order_id: int, request: Request, db: Session = Depends(get_db)):
    """Render the order confirmation page after a successful checkout.

    For Clickworker study sessions the completion code is **not** shown here:
    the worker is routed onward to the post-session attention check
    (``/study/attention``), and the code is only revealed after that check is
    submitted.
    """
    ctx = _base_context(request, db)
    order = db.query(Order).filter(Order.id == order_id).first()
    if not order:
        return HTMLResponse("Order not found", status_code=404)
    ctx["order"] = order
    return templates.TemplateResponse(request, name="confirmation.html", context=ctx)


# ── Clickworker study: completion flow ──────────────────────────────────────

# Ordinary e-commerce departments that the seeded catalog deliberately does NOT
# carry. The check mixes one real category with these decoys per question.
#
# This replaced a product-name check whose decoys were absurd ("Quantum Toaster
# 9000"), so it could be passed without having looked at the shop at all.
# Category recall is answerable only from the session itself, which is why the
# category links were also removed from the navbar (see base.html).
_ATTENTION_DECOYS = [
    "Books & Media",
    "Garden & Outdoor",
    "Toys & Games",
    "Beauty & Personal Care",
    "Pet Supplies",
    "Automotive",
]


def _completion_code(*, order_id: int | None = None, session_id: str = "") -> str:
    """Generate a deterministic 6-digit study completion code.

    Purchase sessions derive the code from the order id; non-purchase sessions
    derive it from the session id so that every worker gets a unique, valid
    code to submit on the study platform.

    Args:
        order_id: The completed order id (purchase path), if any.
        session_id: The visitor session id (non-purchase path).

    Returns:
        A 6-digit numeric string.
    """
    if order_id is not None:
        return str((order_id * 7919 + 100003) % 900000 + 100000)
    digest = hashlib.sha256((session_id or "anon").encode("utf-8")).hexdigest()
    return str(int(digest, 16) % 900000 + 100000)


def _attention_questions(db: Session, n: int = 2) -> list[dict]:
    """Build ``n`` attention-check questions from the seeded categories.

    Each question presents one genuine shop category mixed with decoy
    departments the shop does not carry. Grading (in :func:`submit_attention`)
    is by membership in the categories table, so the correct answers do not
    need to be smuggled through the form.

    Args:
        db: Active SQLAlchemy database session.
        n: Number of questions to generate.

    Returns:
        A list of dicts with ``name``, ``prompt``, and shuffled ``options``.
    """
    names = [c.name for c in db.query(Category).all()]
    reals = random.sample(names, min(n, len(names))) if names else []
    decoy_pool = random.sample(_ATTENTION_DECOYS, min(len(_ATTENTION_DECOYS), 3 * len(reals)))
    questions: list[dict] = []
    for i, real_name in enumerate(reals):
        options = [real_name] + decoy_pool[i * 3:i * 3 + 3]
        random.shuffle(options)
        questions.append({
            "name": f"q{i + 1}",
            "prompt": "Which of these product categories did the shop have?",
            "options": options,
        })
    return questions


@router.get("/study/attention", response_class=HTMLResponse)
def attention_page(request: Request, db: Session = Depends(get_db), order_id: str = ""):
    """Render the post-session 2-question attention check.

    Args:
        request: Incoming request (study cookies, template).
        db: Injected database session.
        order_id: Optional order id (purchase path) carried through to the
            completion screen so the right code is generated.

    Returns:
        An HTML response rendering ``attention.html``.
    """
    ctx = _base_context(request, db)
    ctx["questions"] = _attention_questions(db)
    ctx["order_id"] = order_id
    return templates.TemplateResponse(request, name="attention.html", context=ctx)


@router.post("/study/attention", response_class=HTMLResponse)
async def submit_attention(request: Request, db: Session = Depends(get_db)):
    """Grade the attention check, log it, and reveal the completion code.

    The worker's answers are graded by checking whether each selected category
    name is a genuine shop category. The result is stored as an
    ``attention_check`` event (stamped with study attribution), and the
    completion code is finally revealed on ``study_complete.html``.

    Args:
        request: Incoming request carrying the submitted form and study cookies.
        db: Injected database session.

    Returns:
        An HTML response rendering ``study_complete.html``.
    """
    form = await request.form()
    real_names = {c.name for c in db.query(Category).all()}

    answers: dict[str, str] = {}
    correct = 0
    total = 0
    for key, value in form.items():
        if key.startswith("q"):
            total += 1
            answers[key] = str(value)
            if str(value) in real_names:
                correct += 1
    passed = total > 0 and correct == total

    sid = _session_id(request)
    worker_id, persona = _study_attribution(request)

    raw_order_id = str(form.get("order_id", "") or "")
    order_id = int(raw_order_id) if raw_order_id.isdigit() else None

    completion_code = _completion_code(order_id=order_id, session_id=sid)
    log_event(
        db,
        session_id=sid,
        event_type="attention_check",
        page="/study/attention",
        metadata={
            "answers": answers, "passed": passed, "correct": correct, "total": total,
            # Persist the revealed code so submissions can be verified later by a
            # simple lookup on worker_id (see Experiments/verify_completion_codes.py).
            "completion_code": completion_code, "order_id": order_id,
        },
        worker_id=worker_id,
        persona=persona,
    )

    ctx = _base_context(request, db)
    ctx["completion_code"] = completion_code
    ctx["passed"] = passed
    return templates.TemplateResponse(request, name="study_complete.html", context=ctx)


@router.get("/study/done", response_class=HTMLResponse)
def end_session_page(request: Request, db: Session = Depends(get_db)):
    """Render the non-purchase end-of-session screen.

    Reached via the always-available "End session" button (see ``base.html``)
    for personas that do not complete a purchase. It routes the worker onward
    to the attention check, after which the completion code is revealed.

    Args:
        request: Incoming request (study cookies, template).
        db: Injected database session.

    Returns:
        An HTML response rendering ``end_session.html``.
    """
    ctx = _base_context(request, db)
    return templates.TemplateResponse(request, name="end_session.html", context=ctx)
