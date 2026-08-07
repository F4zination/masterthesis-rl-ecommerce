# StudyDispatcher

A tiny FastAPI service that is the **single entry point** for the Clickworker /
Prolific study. It randomly but *balancedly* assigns each participant to one of
the **2 × 5 = 10** experiment cells (condition × persona) and redirects their
browser to the matching demo site.

```
participant ──▶ GET https://go.<domain>/go?pid=<participant_id>
                          │  (balance persona, then policy within persona)
                          ▼
         302 https://v2.<domain>/?wid=<pid>&persona=fastbuyer     (condition V2)
   or    302 https://v3.<domain>/?wid=<pid>&persona=explorer      (condition V3)
```

The demo sites need **no changes**: they already read `wid`/`persona` from the
landing URL and persist them into cookies (see `DemoSiteV2/app/routers/shop.py`).

## Why a dispatcher instead of platform-side randomization

The design is a balanced between-subjects 2×5 with a per-cell quota (≥30/cell).
Most crowd platforms only split traffic between *listings*, which makes the 10
cells drift out of balance and can't easily de-duplicate repeat workers. The
dispatcher gives you one link, **guaranteed cell balance**, idempotent
assignment per participant, and a central ledger.

## Behaviour

- **Participant-friendly**: the root page asks for a nickname and explicit
  study/data-processing consent, records the consent version and timestamp,
  creates a unique participant id, and sends the participant into the assignment flow.
- **Blocked and balanced**: each new participant enters a least-filled persona
  (random tie-break), then the least-filled policy arm within that persona
  (random tie-break). Persona totals and within-persona V2/V3 counts stay even.
- **Idempotent**: a repeat `pid` always returns to the same cell — workers can
  reopen their link and resume.
- **Restart-safe**: assignments persist in SQLite (`/app/data/dispatcher.db`).
- **Quota-aware**: once every cell reaches `TARGET_PER_CELL`, new participants
  are sent to `STUDY_FULL_URL` (or shown a "study full" page). Participants who
  already have an assignment are never blocked.
- **One researcher login**: a signed, secure cookie unlocks the combined
  dispatcher dashboard and the V2/V3 analytics dashboards across subdomains.

## Endpoints

| Route | Purpose |
|---|---|
| `GET /` | Show the styled nickname and informed-consent form. |
| `POST /start` | Validate consent, create a unique participant id, assign a cell, and redirect. |
| `GET /imprint` | Provider information/legal notice. |
| `GET /privacy` | Article 13 GDPR privacy and browser-storage information. |
| `GET /participant-information` | Concise study and informed-consent information. |
| `GET /go?pid=<id>` | Assign + redirect. Also accepts `PROLIFIC_PID`, `participant_id`, `worker_id`, `wid`. With no id, mints an anonymous one (handy for manual testing). |
| `GET /admin/login` | Researcher login form. |
| `GET /admin` | Combined V2/V3 statistics and assignment-cell dashboard. |
| `GET /admin/api/overview` | JSON backing the combined dashboard. Requires the admin session or bearer token. |
| `GET /admin/counts` | JSON of per-cell fill counts. Requires the admin session or `Authorization: Bearer <ADMIN_TOKEN>`; the old query token remains compatible. |
| `GET /healthz` | Liveness plus service and legal-configuration flags. |

## Configuration (env vars)

| Var | Required | Default | Meaning |
|---|---|---|---|
| `V2_BASE_URL` | yes | – | Public base URL of condition V2 (e.g. `https://v2.example.com`) |
| `V3_BASE_URL` | yes | – | Public base URL of condition V3 |
| `TARGET_PER_CELL` | no | `30` | Recruitment target per cell |
| `ADMIN_TOKEN` | yes (deployment) | – | Independent bearer token for scripted admin API access |
| `ADMIN_USERNAME` | no | `admin` | Researcher login username |
| `ADMIN_PASSWORD` | yes (deployment) | `ADMIN_TOKEN` | Researcher login password; use an independent random value |
| `ADMIN_SESSION_SECRET` | yes (deployment) | `ADMIN_TOKEN` | HMAC key shared with V2/V3 for signed login cookies |
| `ADMIN_COOKIE_DOMAIN` | yes (subdomains) | – | Parent domain, e.g. `.thedemoshop.live` |
| `ADMIN_SESSION_TTL_SECONDS` | no | `28800` | Login lifetime in seconds (eight hours by default) |
| `V2_ANALYTICS_INTERNAL_URL` | no | `V2_BASE_URL` | Internal V2 address used by the combined dashboard |
| `V3_ANALYTICS_INTERNAL_URL` | no | `V3_BASE_URL` | Internal V3 address used by the combined dashboard |
| `STUDY_FULL_URL` | no | – | Redirect target when the study is full |
| `DISPATCHER_DB_PATH` | no | `/app/data/dispatcher.db` | Assignment ledger location |

The deployment must also configure the `LEGAL_*`, `STUDY_RESEARCHER_NAME`,
`STUDY_INSTITUTION`, and `STUDY_RETENTION_PERIOD` values shown in
[`../.env.example`](../.env.example). Missing mandatory legal fields are visibly
flagged on the legal-notice page rather than being filled with invented details.
The direct `/go` integration route does not itself collect consent; only use it
where the recruitment platform has already supplied the same participant/privacy
information and retains a valid consent record.

> Run with a **single worker** (the Dockerfile CMD does). The balance lock is
> in-process; multiple workers would race on cell selection.

## Run locally (without Docker)

```bash
cd StudyDispatcher
pip install -r requirements.txt
V2_BASE_URL=http://127.0.0.1:8002 V3_BASE_URL=http://127.0.0.1:8003 \
  ADMIN_TOKEN=dev ADMIN_PASSWORD=dev ADMIN_SESSION_SECRET=local-secret \
  ADMIN_COOKIE_SECURE=false DISPATCHER_DB_PATH=./data/dispatcher.db \
  uvicorn app.main:app --port 8090
# then: curl -i "http://127.0.0.1:8090/go?pid=TEST1"
```

## Deployment

The dispatcher is one service in the top-level `docker-compose.yml` stack
(dispatcher + V2 + V3 + nginx/TLS). See [../deploy/README.md](../deploy/README.md).
