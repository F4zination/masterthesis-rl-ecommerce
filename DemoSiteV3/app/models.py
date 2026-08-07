from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, ForeignKey, JSON
)
from sqlalchemy.orm import relationship
from .database import Base


class Category(Base):
    """Product category grouping related items together.

    Attributes:
        id: Primary key.
        name: Human-readable category name (e.g., "Electronics").
        slug: URL-friendly identifier used in route paths (e.g., "electronics").
        description: Optional long-form description of the category.
        products: Related products belonging to this category.
    """

    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text, default="")

    products = relationship("Product", back_populates="category")


class Product(Base):
    """A single purchasable item in the store.

    Attributes:
        id: Primary key.
        name: Display name of the product.
        slug: URL-friendly identifier used in route paths.
        description: Detailed product description.
        price: Unit price in the store currency (floating point).
        image_url: URL pointing to the product image.
        category_id: Foreign key reference to the parent Category.
        stock: Available inventory count (defaults to 100).
        category: Related Category instance (lazy-loaded relationship).
    """

    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    slug = Column(String(200), unique=True, nullable=False, index=True)
    description = Column(Text, default="")
    price = Column(Float, nullable=False)
    image_url = Column(String(500), default="")
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=False)
    stock = Column(Integer, default=100)

    category = relationship("Category", back_populates="products")


class Event(Base):
    """A single tracked user interaction event.

    Events capture any meaningful action a visitor performs (e.g., page
    views, clicks, scroll depth) and are used to feed the bandit reward
    signal via :func:`~app.services.decision.apply_reward_from_event`.

    Attributes:
        id: Primary key.
        session_id: Anonymous session identifier for the visitor.
        event_type: Semantic label for the interaction (e.g., ``"add_to_cart"``).
        page: URL path where the event occurred.
        element: Optional CSS selector or name of the interacted element.
        timestamp: UTC datetime when the event was recorded.
        metadata_json: Arbitrary JSON payload with event-specific data.
        worker_id: Clickworker study participant id (from the ``wid`` URL
            param / cookie). Empty for organic, non-study traffic.
        persona: Clickworker study persona/scenario assignment (from the
            ``persona`` URL param / cookie). Empty for non-study traffic.
    """

    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    page = Column(String(500), default="")
    element = Column(String(200), default="")
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(JSON, default=dict)
    worker_id = Column(String(64), default="", index=True)
    persona = Column(String(40), default="", index=True)


class CartItem(Base):
    """A single line item in a visitor's shopping cart.

    Cart items are scoped to an anonymous session and persist until the
    session's cart is cleared (e.g., after checkout).

    Attributes:
        id: Primary key.
        session_id: Anonymous session identifier.
        product_id: Foreign key reference to the associated Product.
        quantity: Number of units added (defaults to 1).
        product: Related Product instance.
    """

    __tablename__ = "cart_items"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), nullable=False, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    quantity = Column(Integer, default=1)

    product = relationship("Product")


class Order(Base):
    """A completed purchase order.

    Created when a visitor submits the checkout form.  The cart is cleared
    immediately after the order is persisted.

    Attributes:
        id: Primary key.
        session_id: Anonymous session identifier of the buyer.
        customer_name: Optional name provided at checkout.
        customer_email: Optional e-mail address provided at checkout.
        total: Final charged amount after any discounts.
        created_at: UTC datetime when the order was placed.
        items: Related OrderItem instances for this order.
        worker_id: Clickworker study participant id (study attribution).
        persona: Clickworker study persona/scenario assignment.
    """

    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), nullable=False, index=True)
    customer_name = Column(String(200), default="")
    customer_email = Column(String(200), default="")
    total = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    worker_id = Column(String(64), default="")
    persona = Column(String(40), default="")

    items = relationship("OrderItem", back_populates="order")


class OrderItem(Base):
    """A single product line within a completed order.

    Records a snapshot of the product price at the time of purchase so
    that historical order totals remain accurate even if prices change.

    Attributes:
        id: Primary key.
        order_id: Foreign key reference to the parent Order.
        product_id: Foreign key reference to the purchased Product.
        quantity: Number of units purchased.
        price: Unit price captured at the time of checkout.
        order: Related Order instance.
        product: Related Product instance.
    """

    __tablename__ = "order_items"

    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)

    order = relationship("Order", back_populates="items")
    product = relationship("Product")


class DecisionLog(Base):
    """Audit record for every bandit decision made during a session.

    Each row captures which action the epsilon-greedy bandit selected for
    a visitor at a specific point in the funnel, together with the full
    context used to make that decision.  Downstream reward attribution
    queries this table to update :class:`BanditArmStat`.

    Attributes:
        id: Primary key.
        session_id: Anonymous session identifier.
        decision_point: Funnel location identifier (e.g., ``"pdp"``, ``"cart"``).
        action: The widget action that was served (e.g., ``"discount_banner"``).
        propensity: The probability assigned to the selected action by the policy.
        context_json: Full context payload including raw/normalised features,
            eligible actions, model name, and per-arm value estimates.
        timestamp: UTC datetime when the decision was made.
    """

    __tablename__ = "decision_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), nullable=False, index=True)
    decision_point = Column(String(50), nullable=False, index=True)
    action = Column(String(50), nullable=False)
    propensity = Column(Float, nullable=False)
    context_json = Column(JSON, default=dict)
    timestamp = Column(DateTime, default=datetime.utcnow)


class SessionDiscount(Base):
    """A discount awarded to a specific visitor session.

    At most one discount record exists per session (enforced by the unique
    constraint on ``session_id``).

    Attributes:
        id: Primary key.
        session_id: Anonymous session identifier (unique).
        discount_pct: Percentage discount to apply (e.g., ``10.0`` for 10 %).
        code: Promo code string associated with the discount.
        applied_at: UTC datetime when the discount was granted.
    """

    __tablename__ = "session_discounts"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), unique=True, nullable=False, index=True)
    discount_pct = Column(Float, default=0.0)
    code = Column(String(50), default="")
    applied_at = Column(DateTime, default=datetime.utcnow)


class BanditArmStat(Base):
    """Accumulated statistics for a single contextual bandit arm.

    An arm is uniquely identified by the combination of
    ``(decision_point, context_key, action)``.  The epsilon-greedy policy
    uses the smoothed mean of ``reward_sum / impressions`` (with a
    Bayesian prior) to rank arms during exploitation.

    Attributes:
        id: Primary key.
        decision_point: Funnel location identifier (e.g., ``"landing"``).
        context_key: Pipe-delimited discrete feature string that identifies
            the context bucket (device, traffic source, numeric buckets).
            The exact fields present depend on ``schema_version``.
        action: The widget action this arm represents.
        impressions: Total number of times this arm has been served.
        reward_sum: Cumulative sum of all rewards (and costs) recorded for
            this arm.
        updated_at: UTC datetime of the last impression or reward update.
        schema_version: Version of the context feature schema used when this
            arm was created.  ``1`` = original flat schema (all features for
            all decision points); ``2`` = per-decision-point feature subsets.
    """

    __tablename__ = "bandit_arm_stats"

    id = Column(Integer, primary_key=True, index=True)
    decision_point = Column(String(50), nullable=False, index=True)
    context_key = Column(String(200), nullable=False, index=True)
    action = Column(String(50), nullable=False, index=True)
    impressions = Column(Integer, default=0)
    reward_sum = Column(Float, default=0.0)
    updated_at = Column(DateTime, default=datetime.utcnow)
    schema_version = Column(Integer, default=2, nullable=False)
