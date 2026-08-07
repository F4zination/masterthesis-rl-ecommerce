# DemoShop - Phase 3 Transition (PPO-First + Offline Fallback)

DemoShop is a FastAPI-based demo e-commerce site used for your thesis pipeline.
The current version implements a hybrid serving approach:

- **PPO policy first** at inference time when a checkpoint is available.
- **Offline-trained policy fallback** when PPO cannot decide or is unavailable.
- **Contextual epsilon-greedy bandit fallback** when neither learned policy can decide.
- **Timing gate + opportunity loop** so the agent can decide **when** to show and **which** widget to show.
- Backward-compatible logging (`decision_logs`, `events`, `bandit_arm_stats`) for reproducible offline RL/OPE workflows.

Experiment specification for clickworker-based policy comparison:
- [../Experiments/Clickworker_Bandit_vs_RL_Experiment.md](../Experiments/Clickworker_Bandit_vs_RL_Experiment.md)

## What Is Implemented

- Product catalog (300 products, 50 per category) with matching images via loremflickr.
- Category pages, product detail pages (PDP), cart, checkout, and confirmation flow.
- Session-based cart API (`/api/cart/*`) with per-session discount management.
- Automatic behavioral event tracking (`/api/events`).
- Hybrid decision runtime (`/api/decision`): PPO first, then offline policy, then bandit fallback.
- Online reward updates from event stream.
- Propensity logging for offline evaluation (IPS/DR style workflows).
- Interactive bandit widgets: trending product carousel, discount banner, scroll-triggered help popup.
- Timing-aware opportunity scheduling on the frontend (deferred retries via `next_check_after_ms`).
- Startup policy health checks with explicit fallback logging.
- Background learner (optional) that retrains PPO policy artifacts online on a fixed interval.
- Versioned policy publishing with active pointer (`active_version.txt`) and hot-reload aware serving.
- Analytics dashboard at `/analytics` with live decision, policy source, and timing statistics.

## How the Hybrid Agent Works

At each opportunity, the frontend calls `POST /api/decision` with:

- `decision_point` (e.g., `cart`)
- contextual features (device/referrer/value buckets)
- opportunity metadata (`opportunity_id`, `opportunity_type`, `opportunity_index`, `elapsed_ms`)

The backend evaluates a timing gate first:

- session-level opportunity cap
- cooldown window between decisions
- optional client-provided earliest-eligible timing

If gated, the API returns:

- `should_show=false`
- `action="no-op"`
- `next_check_after_ms`
- `reason_code`

If eligible, serving policy runs:

1. **PPO policy** (`ppo_policy.pt`) when the checkpoint is available and valid.
2. **Offline policy** (`trained_policy.json`) when PPO cannot decide or is unavailable.
3. **Bandit fallback** when neither learned policy can decide.
4. **Fail-closed PPO serving** in `ppo_only` mode; `offline_only` retains its explicit no-op fallback.

All served decisions are logged with policy/timing metadata in `DecisionLog.context_json`.

## Background Learner (Online Policy Updates)

The app can run an in-process background learner thread (disabled by default) that performs:

1. Windowed extraction from recent `decision_logs` + `events`.
2. Optional bootstrap with `CustomerSimulation/output/offline_transitions.jsonl` for PPO pretraining.
3. Training via `OfflineTraining/train_ppo_policy.py` when `LEARNER_ALGO=ppo`.
4. Lightweight artifact validation.
5. Versioned publish to `OfflineTraining/outputs/vXXXXXX/`.
6. Active pointer update in `active_version.txt` plus checkpoint publish to `ppo_policy.pt`.

Serving resolves the active version first and falls back safely if no valid artifact is available.

### Learner Cycle

- Trigger: startup + every `LEARNER_INTERVAL_SECONDS`.
- Gate: skip cycle if recent decision volume is below `LEARNER_MIN_DECISIONS`.
- Window: `now - LEARNER_LOOKBACK_HOURS` to `now`.
- Timeout: bounded by `LEARNER_MAX_TRAIN_MINUTES`.
- Output metadata: `learner_run.json` stored with each version.

## Bandit Fallback Details

When fallback policy is used, action selection is epsilon-greedy:

- With probability **ε = 0.20** — pick a random action (*explore*).
- Otherwise — pick the action with the highest estimated mean reward for this context (*exploit*).

The estimate for an arm is a Bayesian-style average over observed rewards, initialised with a weak prior (5 pseudo-counts at 0.0) so every arm starts equal and unknown arms are not penalised:

```
mean_estimate = (reward_sum + prior_count × prior_mean) / (impressions + prior_count)
```

Every time an action is shown, its impression counter is incremented and a small **action cost** is subtracted immediately (to discourage showing widgets when doing nothing is fine).

When a user then does something — clicks a widget, adds to cart, reaches checkout, purchases — an event is fired to `POST /api/events`. The backend traces that event back to the most recent decision for this session (using the `decision_id` passed in metadata when available) and adds reward to the selected arm.

Over time, arms with consistently good outcomes accumulate higher reward sums and are chosen more often. Exploration keeps collecting data from sub-optimal arms so the policy can recover if circumstances change.

All decisions are written to `decision_logs` with propensity, context, eligible actions, model, `policy_source`, and timing metadata — enabling **inverse-propensity-weighted (IPS) off-policy evaluation** and transition extraction.

## Decision Design

### Decision Points

| Decision point | Trigger | Actions |
|:---|:---|:---|
| `landing` | Home page slot opportunity | All configured actions |
| `pdp` | Product detail slot opportunity | All configured actions |
| `cart` | Cart slot opportunity | All configured actions |
| `checkout` | Checkout slot opportunity | All configured actions |
| `scroll_engagement` | User scrolls past 70 % | All configured actions |

The frontend opportunity loop can defer and retry requests when the server returns `should_show=false` with `next_check_after_ms`.

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

- Policy modes: `offline_first` (default), `offline_only`, or bandit fallback behavior.
- Logged per decision: selected action, propensity, eligible actions, normalized context, model, `policy_source`, timing gate metadata, and opportunity metadata.
- Arm statistics stored in `bandit_arm_stats` and updated online.

### Opportunity Lifecycle Events

The frontend emits additional events through `/api/events`:

- `opportunity_checked`
- `opportunity_skipped`
- `opportunity_served`

These power timing-outcome analytics without breaking legacy reward attribution.

### Runtime Configuration

Environment variables (from `app/config.py`):

- `POLICY_MODE` (default: `ppo_first`)
- `REQUIRE_PPO_CHECKPOINT` (default: `false`; set `true` with `POLICY_MODE=ppo_only` for the frozen study so startup fails if the checkpoint is unavailable)
- `OFFLINE_POLICY_PATH` (default: `../OfflineTraining/outputs/trained_policy.json`)
- `ACTIVE_POLICY_POINTER_PATH` (default: `../OfflineTraining/outputs/active_version.txt`)
- `TIMING_ENABLED` (default: `true`)
- `MAX_OPPORTUNITIES_PER_SESSION` (default: `30`)
- `MIN_DECISION_COOLDOWN_MS` (default: `0`)
- `OFFLINE_POLICY_DEFER_MS_DEFAULT` (default: `4000`)
- `ANALYTICS_TIMELINE_LOOKBACK_HOURS` (default: `24`)
	Controls the rolling lookback window for the “Decisions over time” chart on `/analytics`.

Learner-specific variables:

- `LEARNER_ENABLED` (default: `true`)
- `LEARNER_INTERVAL_SECONDS` (default: `21600`, 6h)
- `LEARNER_MIN_DECISIONS` (default: `50`)
- `LEARNER_LOOKBACK_HOURS` (default: `24`)
- `LEARNER_MAX_TRAIN_MINUTES` (default: `20`)
- `LEARNER_TIMING_MODE` (default: `opportunity`)
- `LEARNER_OUTPUT_DIR` (default: `../OfflineTraining/outputs`)
- `LEARNER_ALGO` (default: `ppo`)

PPO-specific variables:

- `PPO_CHECKPOINT_PATH` (default: `../OfflineTraining/outputs/ppo_policy.pt`)
- `PPO_PRETRAIN_DATASET_PATH` (default: `../CustomerSimulation/output/offline_transitions.jsonl`)
- `PPO_HIDDEN_SIZES` (default: `128,64`)
- `PPO_GAMMA` (default: `0.95`)
- `PPO_CLIP_EPS` (default: `0.2`)
- `PPO_ENTROPY_COEF` (default: `0.01`)
- `PPO_VALUE_COEF` (default: `0.5`)
- `PPO_LR` (default: `3e-4`)
- `PPO_GAE_LAMBDA` (default: `0.95`)
- `PPO_EPOCHS` (default: `6`)
- `PPO_MINIBATCH_SIZE` (default: `64`)
- `PPO_MAX_GRAD_NORM` (default: `0.5`)

At startup, policy health is checked and PPO/offline fallback mode is logged clearly in app logs. The frozen-study deployment uses `POLICY_MODE=ppo_only` plus `REQUIRE_PPO_CHECKPOINT=true`, which rejects a missing or invalid PPO artifact. If PPO inference cannot produce an eligible action after startup, the request fails instead of silently changing the V3 treatment.

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
| `help_popup_click` | +1.0 *(legacy — never emitted; help-popup interactions are rewarded as `widget_click`)* |
| `scroll_depth` ≥ 50 % | +0.2 *(emitted at most once per page view)* |
| `dwell_time` ≥ 30 s | +0.2 *(emitted at most once per page view)* |

Action costs are subtracted on impression to discourage over-intervention.
Per decision the frontend emits at most **one** widget interaction event
(`widget_click` XOR `widget_dismiss`, whichever happens first), matching the
simulator's mutually exclusive one-shot click/dismiss sampling.

## Session Discounts

A `SessionDiscount` table stores one discount row per session. The `POST /api/cart/discount` endpoint applies a 10 % discount (code `SAVE10`) once per session. The discount is cleared automatically after a successful checkout. Cart and checkout pages display a Subtotal / Discount / Total breakdown when a discount is active.

## API Endpoints

| Endpoint | Method | Purpose |
|:---|:---|:---|
| `/api/decision` | POST | Request decision with timing-aware response (`should_show`, `next_check_after_ms`, `reason_code`) |
| `/api/events` | POST | Log event and trigger reward update |
| `/api/cart` | GET | Cart snapshot with subtotal, discount, and total |
| `/api/cart/add` | POST | Add item to cart |
| `/api/cart/update` | POST | Update item quantity |
| `/api/cart/remove` | POST | Remove item |
| `/api/cart/discount` | GET | Check current session discount |
| `/api/cart/discount` | POST | Apply SAVE10 (10 %) discount to session |
| `/api/trending` | GET | 6 random products for the trending carousel |
| `/api/frequently_bought_together` | GET | Products frequently purchased together (falls back to trending) |
| `/analytics` | GET | Analytics dashboard with decision timeline, policy source breakdown, and PPO checkpoint health panel |
| `/api/analytics/policy_breakdown` | GET | Decision counts by policy source/model/decision point |
| `/api/analytics/timing_outcomes` | GET | Opportunity checked/skipped/served metrics and defer behavior |
| `/api/analytics/learner_status` | GET | Background learner runtime status, last-cycle metadata, and PPO checkpoint health (`available`, `valid`, vocab dims) |

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
cd DemoSiteV3
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

1. Start app and confirm startup logs show policy health + fallback mode.
2. Open homepage/PDP/cart/checkout and confirm widgets still render normally.
3. Trigger scroll engagement and verify help popup behavior remains intact.
4. Inspect `/api/decision` responses in devtools: verify `should_show`, `reason_code`, and `next_check_after_ms` are present.
5. Interact with widgets and cart flow; confirm rewards still update.
6. Open `/analytics` and inspect policy source breakdown and timing outcomes sections.
7. Inspect `decision_logs` context JSON for `policy_source`, `timing_gate`, and `opportunity` fields.
8. (Optional) Move/rename offline policy artifact and verify startup fallback warning appears.

### Validate Online Learner

1. Set `LEARNER_ENABLED=true` and start the app.
2. Check startup logs for `Background learner started`.
3. Generate traffic in the shop so decisions/events accumulate.
4. Call `/api/analytics/learner_status` and verify cycle status transitions (`running`, `succeeded`, `skipped`, or `failed`).
5. Verify `OfflineTraining/outputs/active_version.txt` points to a published version directory.
6. Verify each published version contains `trained_policy.json`, `ppo_policy.pt`, and `learner_run.json`.
7. Confirm decisions continue serving during learner runs (no request-path interruption).
8. Confirm `CustomerSimulation/output/offline_transitions.jsonl` is being used as a PPO bootstrap dataset when no checkpoint exists yet.

## Roadmap

- Expand opportunity triggers beyond page slot + single scroll threshold.
- Add richer sequential state features and long-horizon objective shaping.
