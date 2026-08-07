# DemoShop - Phase 2 (Contextual Bandits)

DemoShop is a FastAPI-based demo e-commerce site used as the Phase 2 implementation of your thesis pipeline.
The core idea is: at each decision point, choose a personalization action for the current context, log propensity, then update online from observed rewards.

## What Is Implemented

- Product catalog (300 products, 50 per category) with matching images via loremflickr.
- Category pages, product detail pages (PDP), cart, checkout, and confirmation flow.
- Session-based cart API (`/api/cart/*`) with per-session discount management.
- Automatic behavioral event tracking (`/api/events`).
- Contextual epsilon-greedy bandit policy (`/api/decision`).
- Online reward updates from event stream.
- Propensity logging for offline evaluation (IPS/DR style workflows).
- Interactive bandit widgets: trending product carousel, discount banner, scroll-triggered help popup.
- Analytics dashboard at `/analytics` with live decision and reward statistics.

## How the Bandit Works

At every decision point, the frontend calls `POST /api/decision` with the current decision point (e.g. `cart`) and a small context object (device, referrer, cart value, etc.).

The backend selects an action using **epsilon-greedy**:

- With probability **ε = 0.20** — pick a random action (*explore*).
- Otherwise — pick the action with the highest estimated mean reward for this context (*exploit*).

The estimate for an arm is a Bayesian-style average over observed rewards, initialised with a weak prior (5 pseudo-counts at 0.0) so every arm starts equal and unknown arms are not penalised:

```
mean_estimate = (reward_sum + prior_count × prior_mean) / (impressions + prior_count)
```

Every time an action is shown, its impression counter is incremented and a small **action cost** is subtracted immediately (to discourage showing widgets when doing nothing is fine).

When a user then does something — clicks a widget, adds to cart, reaches checkout, purchases — an event is fired to `POST /api/events`. The backend traces that event back to the most recent decision for this session (using the `decision_id` passed in the event metadata) and adds the corresponding reward to `reward_sum` for that arm. This is an **online incremental update**: no batch training step required.

Over time, arms with consistently good outcomes accumulate higher reward sums and are chosen more often. The explore fraction (20 %) keeps collecting data from sub-optimal arms so the policy can recover if circumstances change.

All decisions are written to `decision_logs` with the full propensity, context, eligible actions, and arm estimates at the time of decision. Logs from the normal exploratory mode can support propensity-based evaluation when overlap and the other OPE assumptions hold.

### Frozen Clickworker mode

The human study does not use online learning. With `FREEZE_POLICY=true`, V2
serves the deterministic greedy action with a stable tie-break, records
propensity 1.0, and does not insert or update `bandit_arm_stats` rows. Event and
decision logs are still written. Consequently, the study logs do not have
alternative-action overlap and are not used for IPS/SNIPS/DR.

Set `REQUIRE_BANDIT_POLICY=true` in a study deployment. Startup then verifies,
before schema creation or product seeding, that the mounted SQLite database is
readable and contains trained bandit rows. The release manifest separately
checks the exact database hash, build provenance, and serving diagnostics.

## Bandit Design

### Decision Points

| Decision point | Trigger | Actions |
|:---|:---|:---|
| `landing` | Home page load | `no-op`, `trending_carousel`, `discount_banner` |
| `pdp` | Product detail page load | `no-op`, `frequently_bought_together`, `discount_banner` |
| `cart` | Cart page load | `no-op`, `discount_banner`, `frequently_bought_together` |
| `checkout` | Checkout page load | `no-op`, `discount_banner`, `trust_badge` |
| `scroll_engagement` | User scrolls past 70 % of the page | `no-op`, `help_popup` |

### Actions

| Action | Description |
|:---|:---|
| `no-op` | No widget shown — the control arm. |
| `trending_carousel` | Horizontally scrollable row of 6 trending products fetched from `/api/trending`. Each card links to the product page. |
| `discount_banner` | Alert with an **Apply Discount** button. Clicking it calls `POST /api/cart/discount` and applies a 10 % SAVE10 discount to the session. |
| `frequently_bought_together` | Horizontally scrollable row of products frequently purchased together, fetched from `/api/frequently_bought_together`. Falls back to trending items if purchase history is insufficient. |
| `trust_badge` | Security/returns reassurance badge image (`static/images/trust_badge.png`) displayed on checkout. |
| `help_popup` | Floating action button (bottom-right `?` icon) that opens a panel with Live Chat, Browse FAQ, and Contact Support options. Every interaction is rewarded. |

### Context Features (Bucketed)

- Device type (from screen width: mobile / tablet / desktop)
- Traffic source (direct / search / social / referral from referrer)
- Page depth (number of pages visited this session)
- Cart total bucket
- Product price bucket (PDP only)
- Cart item count bucket
- Scroll percentage (scroll_engagement point)

### Policy and Logging

- Epsilon-greedy policy with exploration (`epsilon = 0.2`).
- Logged per decision: selected action, propensity, eligible actions, normalized context, and model version.
- Arm statistics stored in `bandit_arm_stats` and updated online.

### Reward Signals

| Event | Reward |
|:---|:---|
| `widget_click` | +1.2 |
| `widget_dismiss` | −0.8 |
| `add_to_cart` | +1.5 |
| `remove_from_cart` | −0.5 |
| `checkout_submit` | +2.0 |
| `purchase` | +8.0 + min(order_total / 100, 4.0) |
| `exit_intent` | −0.6 |
| `help_popup_click` | +1.0 |
| `scroll_depth` ≥ 50 % | +0.2 |
| `dwell_time` ≥ 30 s | +0.2 |

Action costs are subtracted on impression to discourage over-intervention.

## Session Discounts

A `SessionDiscount` table stores one discount row per session. The `POST /api/cart/discount` endpoint applies a 10 % discount (code `SAVE10`) once per session. The discount is cleared automatically after a successful checkout. Cart and checkout pages display a Subtotal / Discount / Total breakdown when a discount is active.

## API Endpoints

| Endpoint | Method | Purpose |
|:---|:---|:---|
| `/api/decision` | POST | Request contextual action at decision point |
| `/api/events` | POST | Log event and trigger reward update |
| `/api/cart` | GET | Cart snapshot with subtotal, discount, and total |
| `/api/cart/add` | POST | Add item to cart |
| `/api/cart/update` | POST | Update item quantity |
| `/api/cart/remove` | POST | Remove item |
| `/api/cart/discount` | GET | Check current session discount |
| `/api/cart/discount` | POST | Apply SAVE10 (10 %) discount to session |
| `/api/trending` | GET | 6 random products for the trending carousel |
| `/api/frequently_bought_together` | GET | Products frequently purchased together (falls back to trending) |
| `/analytics` | GET | Analytics dashboard |

In the deployed study, `/analytics` and every `/api/analytics/*` route require
the signed researcher cookie issued by the dispatcher or an
`Authorization: Bearer <ADMIN_TOKEN>` header. Configure `ADMIN_USERNAME`,
`ADMIN_SESSION_SECRET`, `ADMIN_COOKIE_NAME`, and `ADMIN_LOGIN_URL` consistently
across the dispatcher, V2, and V3 services.

## Tech Stack

- Backend: FastAPI, SQLAlchemy, SQLite
- Frontend: Jinja2 templates, Bootstrap 5, Bootstrap Icons, vanilla JavaScript
- Charts: Chart.js 4
- Images: loremflickr.com (`/400/400/{keyword}?lock={N}`)
- Data store: SQLite (`demosite.db`, auto-created)

## Setup

```bash
cd DemoSiteV2
pip install -r requirements.txt
python -m shared_schema.migrations run --db-path ./demosite_test.db
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open http://localhost:8000.

On first run, tables are created automatically and 300 seed products across 6 categories are inserted. To force a reseed (drops and recreates all product data):

```bash
python -m app.seed --force
```

## Database Migrations

Schema migrations are centralized in SharedSchema and versioned.

```bash
# Show current schema version and pending steps
python -m shared_schema.migrations status --db-path ./demosite_test.db

# Run all pending migrations (or to a specific version)
python -m shared_schema.migrations run --db-path ./demosite_test.db
python -m shared_schema.migrations run --db-path ./demosite_test.db --target-version 2
```

Notes:
- Current production schema is v2.
- Legacy v1 arms are retained and tagged with `schema_version = 1`.
- Local `migrate_v2.py` scripts were removed; use the shared migration CLI.

## Quick Validation Flow

1. Open the homepage and a PDP; observe bandit widgets appearing (or `no-op`).
2. Click/dismiss widgets and add/remove cart items.
3. Scroll to the bottom of any page to trigger the `scroll_engagement` decision point — a help FAB should appear.
4. Apply a discount from the `discount_banner` widget and verify the cart total updates.
5. Complete checkout once.
6. Open `/analytics` and inspect the timeline, arm estimates, and action distribution.
7. Inspect `decision_logs`, `events`, and `bandit_arm_stats` in `demosite.db` to verify:
   - propensity is logged per decision,
   - rewards flow from events to selected arms,
   - arm mean estimates change over time.

## Roadmap

- Phase 3: Offline RL training and OPE over logged trajectories.
- Phase 4: Full sequential online RL policy with state transitions and long-term objectives.
