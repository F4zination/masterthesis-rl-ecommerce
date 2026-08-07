#!/usr/bin/env bash
#
# One-time TLS bootstrap for the study stack.
#
# Solves the chicken-and-egg problem: nginx refuses to start without the
# certificate files referenced in its config, but certbot needs nginx running
# to answer the ACME http-01 challenge. We therefore:
#   1. drop in self-signed dummy certs so nginx can boot,
#   2. start nginx,
#   3. delete the dummies and request real certs from Let's Encrypt,
#   4. reload nginx.
#
# Run once from the repo root (Linux host with Docker + Compose v2):
#   ./deploy/init-letsencrypt.sh
#
# Re-running is safe; it re-issues certs (--force-renewal).
#
# Adapted from the canonical wmnnd/nginx-certbot bootstrap.
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is required." >&2
  exit 1
fi
if [ ! -f .env ]; then
  echo "Error: .env not found. Copy .env.example to .env and fill it in." >&2
  exit 1
fi

# Read only simple deployment keys. Do not source the Docker dotenv file:
# legal/study values may legitimately contain unquoted spaces.
dotenv_value() {
  local key="$1"
  local value
  value="$(sed -n "s/^${key}=//p" .env | tail -n 1)"
  value="${value%$'\r'}"
  if [[ "$value" == \"*\" && "$value" == *\" ]]; then
    value="${value:1:${#value}-2}"
  elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
    value="${value:1:${#value}-2}"
  fi
  printf '%s' "$value"
}

domains=(
  "${APEX_DOMAIN:-$(dotenv_value APEX_DOMAIN)}"
  "${DISPATCHER_DOMAIN:-$(dotenv_value DISPATCHER_DOMAIN)}"
  "${V2_DOMAIN:-$(dotenv_value V2_DOMAIN)}"
  "${V3_DOMAIN:-$(dotenv_value V3_DOMAIN)}"
)
email="${CERTBOT_EMAIL:-$(dotenv_value CERTBOT_EMAIL)}"
staging="${CERTBOT_STAGING:-$(dotenv_value CERTBOT_STAGING)}"
data_path="./deploy/certbot"
rsa_key_size=4096

for d in "${domains[@]}"; do
  if [ -z "$d" ] || [[ "$d" == *example.com ]]; then
    echo "Error: set real domains in .env (found '$d')." >&2
    exit 1
  fi
done

mkdir -p "$data_path/conf" "$data_path/www"

# 1. Recommended TLS options + DH params (referenced by the nginx config).
if [ ! -e "$data_path/conf/options-ssl-nginx.conf" ] || [ ! -e "$data_path/conf/ssl-dhparams.pem" ]; then
  echo "### Downloading recommended TLS parameters ..."
  curl -fsSL https://raw.githubusercontent.com/certbot/certbot/main/certbot-nginx/certbot_nginx/_internal/tls_configs/options-ssl-nginx.conf \
    > "$data_path/conf/options-ssl-nginx.conf"
  curl -fsSL https://raw.githubusercontent.com/certbot/certbot/main/certbot/certbot/ssl-dhparams.pem \
    > "$data_path/conf/ssl-dhparams.pem"
fi

# 2. Dummy self-signed certs so nginx can start.
for domain in "${domains[@]}"; do
  echo "### Creating dummy certificate for $domain ..."
  live="/etc/letsencrypt/live/$domain"
  mkdir -p "$data_path/conf/live/$domain"
  docker compose run --rm --entrypoint "\
    openssl req -x509 -nodes -newkey rsa:$rsa_key_size -days 1 \
      -keyout '$live/privkey.pem' -out '$live/fullchain.pem' -subj '/CN=localhost'" certbot
done

# 3. Start nginx with the dummy certs.
echo "### Starting nginx ..."
docker compose up --force-recreate -d nginx

# 4. Delete dummies and request the real certificates.
for domain in "${domains[@]}"; do
  echo "### Deleting dummy certificate for $domain ..."
  docker compose run --rm --entrypoint "\
    rm -Rf /etc/letsencrypt/live/$domain && \
    rm -Rf /etc/letsencrypt/archive/$domain && \
    rm -Rf /etc/letsencrypt/renewal/$domain.conf" certbot
done

staging_arg=""
if [ "$staging" != "0" ]; then
  echo "### Using Let's Encrypt STAGING environment (CERTBOT_STAGING=$staging)."
  staging_arg="--staging"
fi

email_arg="--register-unsafely-without-email"
if [ -n "$email" ] && [[ "$email" != *example.com ]]; then
  email_arg="--email $email"
fi

for domain in "${domains[@]}"; do
  echo "### Requesting Let's Encrypt certificate for $domain ..."
  docker compose run --rm --entrypoint "\
    certbot certonly --webroot -w /var/www/certbot $staging_arg $email_arg \
      -d $domain --rsa-key-size $rsa_key_size --agree-tos --no-eff-email --force-renewal" certbot
done

# 5. Reload nginx with the real certs.
echo "### Reloading nginx ..."
docker compose exec nginx nginx -s reload

echo
echo "### Done. Bring up the full stack with:  docker compose up -d"
