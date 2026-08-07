# Shared VPS ingress with MaglioSite

This deployment keeps the existing MaglioSite nginx as the only process bound
to host ports 80 and 443. Its existing `elisamaglio.de` configuration is not
replaced. A compose override connects that nginx to the study containers over
the external `study-gateway` network and adds hostname-based study routing.

## Resulting routes

| Request host | Destination |
|---|---|
| `elisamaglio.de`, `www.elisamaglio.de` | Existing MaglioSite `web:3000` service |
| `thedemoshop.live` | `301` to `https://start.thedemoshop.live`, preserving path/query |
| `start.thedemoshop.live` | Study dispatcher |
| `v2.thedemoshop.live` | DemoSiteV2 |
| `v3.thedemoshop.live` | DemoSiteV3 |

## 1. DNS

At the DNS provider for `thedemoshop.live`, create these records with the
VPS's public IPv4 address:

```text
Type  Name   Value
A     @      VPS_IPV4
A     start  VPS_IPV4
A     v2     VPS_IPV4
A     v3     VPS_IPV4
```

Add equivalent AAAA records only if nginx is reachable through the VPS's
public IPv6 address. A stale/wrong AAAA record can make ACME validation fail.

Wait until all four names resolve to the VPS before requesting certificates:

```bash
getent ahostsv4 thedemoshop.live
getent ahostsv4 start.thedemoshop.live
getent ahostsv4 v2.thedemoshop.live
getent ahostsv4 v3.thedemoshop.live
```

## 2. Prepare the study checkout on the VPS

From the study repository:

```bash
cp example.env .env
nano .env
```

At minimum, set the four domains, a real `CERTBOT_EMAIL`, independent strong
values for `ADMIN_TOKEN`, `ADMIN_PASSWORD`, and `ADMIN_SESSION_SECRET`,
`ADMIN_COOKIE_DOMAIN=.thedemoshop.live`, and every legal/study placeholder.
Start with `CERTBOT_STAGING=1` for a non-browser-trusted test certificate.

The frozen treatment artifacts are intentionally not generated during a public
deployment. Prefer copying the exact `ppo_policy.pt`, `trained_policy.json`, and
V2 `demosite.db` selected for the experiment. If no canonical artifacts exist
yet, generate 5,000 seeded history-state sessions and create deterministic
sequential-PPO/V2 defaults before starting the gateway:

```bash
./deploy/prepare-study-artifacts.sh
```

Record the printed SHA-256 values with the study deployment notes. Do not rerun
training after participant collection begins.

The certificate helper assumes the repositories are sibling directories:

```text
/path/to/MasterThesisProject
/path/to/MaglioSite
```

If they are elsewhere, pass the existing website directory explicitly:

```bash
MAGLIOSITE_DIR=/absolute/path/to/MaglioSite \
  ./deploy/shared-gateway/init-letsencrypt.sh
```

Otherwise, run:

```bash
./deploy/shared-gateway/init-letsencrypt.sh
```

The helper creates `study-gateway`, builds/starts only the three study apps
plus certbot, recreates the MaglioSite nginx with the additive override, issues
one certificate covering all four study names, tests nginx, and reloads it.
It never replaces MaglioSite's existing `nginx/default.conf` or Elisa
certificate mount.

## 3. Switch from staging to a trusted certificate

After the staging run succeeds:

```bash
sed -i 's/^CERTBOT_STAGING=.*/CERTBOT_STAGING=0/' .env
FORCE_RENEWAL=1 ./deploy/shared-gateway/init-letsencrypt.sh
```

Use `MAGLIOSITE_DIR=/absolute/path/to/MaglioSite` on the second command too if
the repositories are not siblings.

## 4. Verify

```bash
curl -sI http://thedemoshop.live/test-path?x=1
curl -sI https://thedemoshop.live/test-path?x=1
curl -sI https://start.thedemoshop.live/healthz
curl -sI https://v2.thedemoshop.live/
curl -sI https://v3.thedemoshop.live/
curl -sI https://elisamaglio.de/
```

The first two commands must return a `Location` beginning with
`https://start.thedemoshop.live/test-path?x=1`. Elisa and all three study
services should return their normal application responses.

Researcher access is available at:

```text
https://start.thedemoshop.live/admin/login
```

After login, `/admin` combines assignment counts with live statistics from the
protected V2 and V3 analytics APIs. Verify that anonymous analytics access is
blocked and bearer-token automation still works:

```bash
curl -sI https://v2.thedemoshop.live/analytics
curl -s https://v2.thedemoshop.live/api/analytics/summary

set -a
source .env
set +a
curl -fsS -H "Authorization: Bearer $ADMIN_TOKEN" \
  https://v2.thedemoshop.live/api/analytics/summary
```

The first response redirects to the dispatcher login, the second returns 401,
and the bearer-authenticated request returns JSON.

Inspect the effective shared nginx configuration and logs if needed:

```bash
cd /absolute/path/to/MaglioSite
export STUDY_REPO_PATH=/absolute/path/to/MasterThesisProject
docker compose \
  -f docker-compose.yml \
  -f "$STUDY_REPO_PATH/deploy/shared-gateway/docker-compose.magliosite.yml" \
  exec nginx nginx -T
docker compose \
  -f docker-compose.yml \
  -f "$STUDY_REPO_PATH/deploy/shared-gateway/docker-compose.magliosite.yml" \
  logs --tail=100 nginx
```

## Future updates

Redeploy study application code from the study checkout with:

```bash
cd /root/MasterThesisProject
docker compose -f docker-compose.yml -f docker-compose.vps.yml up -d --build

# Refresh cached Docker upstream addresses after the app containers are recreated.
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

When recreating the MaglioSite nginx, continue including the additive override
and set `STUDY_REPO_PATH`; running its base compose alone would omit the study
configuration on the newly created container. The nginx override reloads TLS
files every six hours, while the study certbot service checks renewal twice a
day.
