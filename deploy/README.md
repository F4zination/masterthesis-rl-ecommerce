# Deployment — Clickworker study stack

`docker-compose.yml` (at the repo root) runs the whole study behind one nginx
reverse proxy with Let's Encrypt TLS:

| Service | Internal | Public URL | Role |
|---|---|---|---|
| redirect only | – | `https://${APEX_DOMAIN}` | Permanent redirect to `https://${DISPATCHER_DOMAIN}` |
| `dispatcher` | `dispatcher:8000` | `https://${DISPATCHER_DOMAIN}` | Balanced cell assignment + redirect |
| `v2` | `v2:8000` | `https://${V2_DOMAIN}` | DemoSiteV2 — Contextual Bandit (frozen) |
| `v3` | `v3:8000` | `https://${V3_DOMAIN}` | DemoSiteV3 — PPO / sequential RL (frozen) |
| `nginx` | – | ports 80/443 | TLS termination + reverse proxy |
| `certbot` | – | – | Issues + auto-renews certificates |

Routing is **subdomain-based** (not path-based) because the demo apps serve
absolute paths (`/`, `/static`, `/api`, `/category/...`); giving each app its
own hostname keeps cookies and asset URLs correct with zero app changes.

## Prerequisites

1. A Linux host with Docker + Compose v2 and ports **80/443** open.
2. **Four DNS records** — `APEX_DOMAIN`, `DISPATCHER_DOMAIN`, `V2_DOMAIN`, and
   `V3_DOMAIN` — each an A/AAAA record pointing at the host's public IP. (For
   example: the apex plus `start.`, `v2.`, and `v3.`.) Only publish an AAAA
   record if the host really accepts inbound IPv6 traffic on ports 80 and 443.
3. The frozen V2 bandit DB built once (so V2 serves a *trained* policy):
   ```bash
   make simulate         # only if CustomerSimulation output doesn't exist yet
   make bandit-policy    # writes DemoSiteV2/data/demosite.db
   ```
   V3's policy is baked into its image from `OfflineTraining/outputs/`.

## First-time bring-up

```bash
cp example.env .env
nano .env                       # set domains, TLS/admin, and all legal fields

# A fresh clone excludes generated policies/databases. Prefer copying the exact
# pre-selected frozen artifacts; otherwise generate history-state trajectories
# and build deterministic sequential-PPO/V2 defaults once.
./deploy/prepare-study-artifacts.sh

# Issue certificates (handles the nginx/certbot chicken-and-egg automatically).
# Keep CERTBOT_STAGING=1 in .env for the first run to avoid rate limits while
# you confirm DNS + ports work; then set it to 0 and re-run for trusted certs.
./deploy/init-letsencrypt.sh

docker compose up -d            # build + start the full stack
docker compose ps               # all healthy?
```

Verify:

```bash
curl -sI  https://${DISPATCHER_DOMAIN}/healthz
curl -sI  "https://${APEX_DOMAIN}/some/path?test=1"  # 301; path/query preserved
curl -sI  https://${V2_DOMAIN}/         # 200 from DemoSiteV2
curl -s   "https://${DISPATCHER_DOMAIN}/go?pid=SMOKE1" -o /dev/null -w '%{redirect_url}\n'
curl -fsS -H "Authorization: Bearer $ADMIN_TOKEN" \
  "https://${DISPATCHER_DOMAIN}/admin/counts"
```

The participant-facing nickname and consent entry page is:
```
https://${DISPATCHER_DOMAIN}/
```

The researcher login and combined V2/V3 dashboard is:

```
https://${DISPATCHER_DOMAIN}/admin/login
```

The login creates a secure, HttpOnly cookie for the configured parent domain,
so links from the combined dashboard open the protected V2 and V3 dashboards
without asking for a second password. Direct unauthenticated requests to
`/analytics` redirect to this login; unauthenticated `/api/analytics/*`
requests return HTTP 401.

For a platform flow that already provides approved participant information and
records consent independently, the legacy participant-id link is:
```
https://${DISPATCHER_DOMAIN}/go?pid={{%PROLIFIC_PID%}}
```
(use the platform's participant-id placeholder; Prolific = `{{%PROLIFIC_PID%}}`).
Do not use this direct route as a substitute for informed consent.

## TLS / certificate notes

- `init-letsencrypt.sh` drops in temporary self-signed certs so nginx can boot,
  then swaps them for real Let's Encrypt certs via the http-01 webroot challenge
  (`/.well-known/acme-challenge/` is served from `deploy/certbot/www`).
- The `certbot` service runs `certbot renew` every 12h; the `nginx` service
  reloads every 6h, so renewed certs are picked up with no manual step.
- Certificate state lives in `deploy/certbot/` (git-ignored).
- Going from staging → production: set `CERTBOT_STAGING=0` in `.env`, then
  `./deploy/init-letsencrypt.sh` again (it force-renews).

## Operations

```bash
docker compose logs -f nginx dispatcher        # tail logs
docker compose up -d --build dispatcher        # redeploy one service
docker compose down                            # stop (keeps volumes/data)
```

To add or rotate admin credentials, generate independent random values and put
them in `.env`:

```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'  # ADMIN_TOKEN
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'  # ADMIN_PASSWORD
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'  # ADMIN_SESSION_SECRET
```

For this deployment, set `ADMIN_COOKIE_DOMAIN=.thedemoshop.live`, then rebuild
the three application services. The shared nginx configuration file does not
change, but reload nginx afterward so it resolves the recreated container IPs:

```bash
cd /root/MasterThesisProject
docker compose -f docker-compose.yml -f docker-compose.vps.yml \
  up -d --build v2 v3 dispatcher

cd /root/MaglioSite
export STUDY_REPO_PATH=/root/MasterThesisProject
docker compose \
  -f docker-compose.yml \
  -f "$STUDY_REPO_PATH/deploy/shared-gateway/docker-compose.magliosite.yml" \
  exec nginx nginx -t
docker compose \
  -f docker-compose.yml \
  -f "$STUDY_REPO_PATH/deploy/shared-gateway/docker-compose.magliosite.yml" \
  exec nginx nginx -s reload
```

Collected study data persists on the host via bind mounts:
`DemoSiteV2/data/`, `DemoSiteV3/data/` (SQLite DBs with `events`, `orders`,
`decision_logs` carrying `worker_id`/`persona`) and `StudyDispatcher/data/`
(the assignment ledger — your ground-truth cell map).

## Co-hosting behind the MaglioSite nginx

Only one container can publish host ports 80 and 443. On a VPS where
MaglioSite already owns those ports, use it as the shared ingress:

1. Create the external `study-gateway` Docker network once.
2. Start this stack with `docker-compose.vps.yml`; its nginx is disabled and
   the three application services join that network under unambiguous aliases.
3. Start MaglioSite with
   `deploy/shared-gateway/docker-compose.magliosite.yml`; its existing nginx
   joins the network and loads the study virtual-host template.
4. Issue the study certificate with
   `deploy/shared-gateway/init-letsencrypt.sh`.

The existing `elisamaglio.de` virtual host stays unchanged. Hostname-based
routing sends `elisamaglio.de` to MaglioSite, the three study subdomains to
their matching services, and the study apex to `start.`. Do not route the
study apps by URL path: they use absolute paths and require separate hosts.

See `deploy/shared-gateway/README.md` for the exact VPS commands.

## Frozen-policy guarantees (already wired in compose)

- `v2`: `FREEZE_POLICY=true`
- `v3`: `FREEZE_POLICY=true`, `LEARNER_ENABLED=false`, `POLICY_MODE=ppo_first`

These match the `ClickworkerExperiment.md` spec so neither policy learns from
participant sessions during the A/B test.
