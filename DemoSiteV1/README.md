# DemoShop – Phase 1

A simple e-commerce shop built as the Phase 1 testground for the RL pipeline described in `Concept/TheoreticalPipeline.md`.

## Features

- **Product Catalog**: ~50 dummy products across 6 categories
- **Shopping Cart**: Session-based add/update/remove
- **Checkout Flow**: Full journey from browsing to order confirmation
- **Event Tracking**: Automatic collection of page views, scroll depth, dwell time, clicks, and exit intent
- **Decision Points**: Placeholder hooks at landing page, product detail, and cart for future bandit/RL widget injection
- **Decision Logging**: All decision-point calls logged with propensity scores for off-policy evaluation

## Tech Stack

| Layer | Technology |
|:------|:-----------|
| Backend | FastAPI + SQLAlchemy (ORM) |
| Database | SQLite (auto-created on first run) |
| Frontend | Jinja2 (SSR templates) + Bootstrap 5 + vanilla JavaScript |
| Session | Cookie-based `session_id` (no authentication) |

## Setup

```bash
cd DemoSite
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The database is automatically created and seeded on first startup. Open http://localhost:8000 in your browser.

---

## Architecture

### High-Level Request Flow

```
Browser
 ├── tracking.js  ──POST /api/events──►  events router  ──►  tracking.log_event()  ──►  events table
 ├── decision.js  ──POST /api/decision─►  decision router ──►  decision.get_decision() ──►  decision_logs table
 └── cart.js      ──POST /api/cart/*──►  api router     ──►  cart service           ──►  cart_items table

Server-Side Rendered Pages (Jinja2)
 GET  /  |  /category/:slug  |  /product/:slug  |  /cart  |  /checkout  |  /confirmation/:id
  └── shop router  ──►  direct ORM queries  ──►  Category / Product / Order / OrderItem tables
```

---

### Project Structure

```
app/
├── main.py              # FastAPI app factory, lifespan (DB init + seed), middleware, router registration
├── config.py            # DATABASE_URL, APP_NAME, SECRET_KEY from environment / defaults
├── database.py          # SQLAlchemy engine, SessionLocal, Base, get_db() dependency, init_db()
├── models.py            # ORM table definitions (see Data Models below)
├── seed.py              # One-time population of categories and products
├── routers/
│   ├── shop.py          # HTML page routes (Jinja2 responses)
│   ├── api.py           # /api/cart/* – cart CRUD
│   ├── events.py        # /api/events – event ingestion
│   └── decision.py      # /api/decision – decision-point serving
├── services/
│   ├── cart.py          # Cart business logic (add / update / remove / total / count)
│   ├── decision.py      # Phase 1 stub: always returns no-op, writes DecisionLog
│   └── tracking.py      # log_event() – persists Event rows
└── static/
    └── js/
        ├── tracking.js  # Automatic behavior tracking (see Event Types below)
        ├── decision.js  # Decision point client: requests action, renders widget, tracks interactions
        └── cart.js      # Async cart operations (add / update / remove)
```

---

### Data Models

| Table | Purpose | Key Columns |
|:------|:--------|:------------|
| `categories` | Product taxonomy | `id`, `name`, `slug` |
| `products` | Product catalog | `id`, `name`, `slug`, `price`, `stock`, `category_id` |
| `events` | Raw user behavior log | `session_id`, `event_type`, `page`, `element`, `timestamp`, `metadata_json` |
| `cart_items` | Live shopping carts | `session_id`, `product_id`, `quantity` |
| `orders` | Completed orders | `session_id`, `customer_name`, `customer_email`, `total`, `created_at` |
| `order_items` | Line items per order | `order_id`, `product_id`, `quantity`, `price` |
| `decision_logs` | Decision-point audit trail | `session_id`, `decision_point`, `action`, `propensity`, `context_json`, `timestamp` |

---

### Page Routes

| Method | Path | Template | Description |
|:-------|:-----|:---------|:------------|
| GET | `/` | `home.html` | Landing page – 8 random featured products |
| GET | `/category/{slug}` | `category.html` | All products in a category |
| GET | `/product/{slug}` | `product.html` | Product detail + 4 related products |
| GET | `/cart` | `cart.html` | Shopping cart |
| GET | `/checkout` | `checkout.html` | Checkout form (redirects to `/cart` if empty) |
| POST | `/checkout` | — | Creates Order, clears cart, redirects to confirmation |
| GET | `/confirmation/{order_id}` | `confirmation.html` | Order success page |

---

### API Endpoints

#### Decision

| Method | Path | Body | Response |
|:-------|:-----|:-----|:---------|
| POST | `/api/decision` | `{session_id, decision_point, context?}` | `{action, propensity, decision_point}` |

#### Events

| Method | Path | Body | Response |
|:-------|:-----|:-----|:---------|
| POST | `/api/events` | `{session_id, event_type, page, element, metadata?}` | `{status, event_id}` |

#### Cart

| Method | Path | Body | Response |
|:-------|:-----|:-----|:---------|
| GET | `/api/cart` | — | `{items[], total, count}` |
| POST | `/api/cart/add` | `{product_id, quantity}` | Updated cart |
| POST | `/api/cart/update` | `{product_id, quantity}` | Updated cart |
| POST | `/api/cart/remove` | `{product_id}` | Updated cart |

All cart endpoints read `session_id` from the request cookie.

---

### Event Types Tracked

Fired automatically by `tracking.js`:

| Event | Trigger |
|:------|:--------|
| `page_view` | Script load |
| `scroll_depth` | 25 / 50 / 75 / 100 % scroll thresholds (de-duped) |
| `dwell_time` | Every 30 s on page + final on `beforeunload` |
| `click` | Any `<a>`, `<button>`, or `[data-track]` element |
| `exit_intent` | Mouse leaves viewport at the top edge (fired once) |

Fired by `decision.js` when a widget is active:

`widget_impression`, `widget_click`, `widget_dismiss`

Fired by `cart.js` on user actions:

`add_to_cart`, `remove_from_cart`

---

### Decision Points

Each decision point maps to a `.widget-slot` element in the Jinja2 template. On page load, `decision.js` POSTs to `/api/decision` and, if a non-no-op action is returned, injects the corresponding widget HTML into the slot.

| Decision Point | Page | Available Actions |
|:---------------|:-----|:-----------------|
| `landing` | `/` | `no-op`, `trending_carousel`, `discount_banner` |
| `pdp` | `/product/{slug}` | `no-op`, `frequently_bought_together`, `discount_banner` |
| `cart` | `/cart` | `no-op`, `discount_banner`, `frequently_bought_together` |

**Phase 1 behaviour:** `services/decision.py::get_decision()` is a stub that always returns `no-op` with `propensity=1.0`. Every call still writes a `DecisionLog` row, ensuring the data infrastructure is exercised from day one.

---

## Future Phases

| Phase | Change | Location |
|:------|:-------|:---------|
| **Phase 2** | Replace `get_decision()` stub with a contextual bandit (epsilon-greedy / Thompson Sampling) | `services/decision.py` |
| **Phase 3** | Offline RL training from `decision_logs` + `events` trajectories | New training module |
| **Phase 4** | Full online RL agent with continuous policy updates | New policy server |

Each phase is a separate folder copy to preserve the baseline.
