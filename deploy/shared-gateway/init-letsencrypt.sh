#!/usr/bin/env bash
# Bootstrap/refresh the study TLS certificate while MaglioSite nginx remains
# the only public ingress on ports 80 and 443.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
study_repo_path="$(cd "$script_dir/../.." && pwd)"
magliosite_dir="${MAGLIOSITE_DIR:-$study_repo_path/../MaglioSite}"
gateway_network="study-gateway"
certificate_name="${STUDY_CERTIFICATE_NAME:-thedemoshop.live}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is required." >&2
  exit 1
fi
if [ ! -f "$study_repo_path/.env" ]; then
  echo "Error: $study_repo_path/.env does not exist." >&2
  echo "Copy example.env to .env and fill in every placeholder first." >&2
  exit 1
fi
if [ ! -f "$magliosite_dir/docker-compose.yml" ]; then
  echo "Error: MaglioSite compose file not found at $magliosite_dir/docker-compose.yml." >&2
  echo "Set MAGLIOSITE_DIR to its absolute directory and run this script again." >&2
  exit 1
fi

required_artifacts=(
  "$study_repo_path/OfflineTraining/outputs/ppo_policy.pt"
  "$study_repo_path/OfflineTraining/outputs/trained_policy.json"
  "$study_repo_path/DemoSiteV2/data/demosite.db"
)
missing_artifacts=()
for artifact in "${required_artifacts[@]}"; do
  if [ ! -s "$artifact" ]; then
    missing_artifacts+=("$artifact")
  fi
done
if [ "${#missing_artifacts[@]}" -gt 0 ]; then
  echo "Error: frozen study artifacts are missing:" >&2
  printf '  %s\n' "${missing_artifacts[@]}" >&2
  echo >&2
  echo "Copy the pre-selected artifacts from the training machine, or run:" >&2
  echo "  ./deploy/prepare-study-artifacts.sh" >&2
  echo "Then run this TLS/deployment helper again." >&2
  exit 1
fi

# Only the simple deployment keys are read here. Unlike `source .env`, this
# accepts Docker dotenv values containing spaces in unrelated legal fields.
dotenv_value() {
  local key="$1"
  local value
  value="$(sed -n "s/^${key}=//p" "$study_repo_path/.env" | tail -n 1)"
  value="${value%$'\r'}"
  if [[ "$value" == \"*\" && "$value" == *\" ]]; then
    value="${value:1:${#value}-2}"
  elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
    value="${value:1:${#value}-2}"
  fi
  printf '%s' "$value"
}

apex_domain="${APEX_DOMAIN:-$(dotenv_value APEX_DOMAIN)}"
dispatcher_domain="${DISPATCHER_DOMAIN:-$(dotenv_value DISPATCHER_DOMAIN)}"
v2_domain="${V2_DOMAIN:-$(dotenv_value V2_DOMAIN)}"
v3_domain="${V3_DOMAIN:-$(dotenv_value V3_DOMAIN)}"
certbot_email="${CERTBOT_EMAIL:-$(dotenv_value CERTBOT_EMAIL)}"
certbot_staging="${CERTBOT_STAGING:-$(dotenv_value CERTBOT_STAGING)}"

domains=("$apex_domain" "$dispatcher_domain" "$v2_domain" "$v3_domain")
for domain in "${domains[@]}"; do
  if [[ ! "$domain" =~ ^[A-Za-z0-9.-]+$ ]] || [[ "$domain" == *example.com ]]; then
    echo "Error: invalid or placeholder study domain: '$domain'." >&2
    exit 1
  fi
done
if [[ ! "$certificate_name" =~ ^[A-Za-z0-9.-]+$ ]]; then
  echo "Error: invalid STUDY_CERTIFICATE_NAME: '$certificate_name'." >&2
  exit 1
fi

study_compose=(
  docker compose
  --project-directory "$study_repo_path"
  -f "$study_repo_path/docker-compose.yml"
  -f "$study_repo_path/docker-compose.vps.yml"
)
gateway_compose=(
  docker compose
  --project-directory "$magliosite_dir"
  -f "$magliosite_dir/docker-compose.yml"
  -f "$script_dir/docker-compose.magliosite.yml"
)

gateway_env=(
  "STUDY_REPO_PATH=$study_repo_path"
  "APEX_DOMAIN=$apex_domain"
  "DISPATCHER_DOMAIN=$dispatcher_domain"
  "V2_DOMAIN=$v2_domain"
  "V3_DOMAIN=$v3_domain"
  "STUDY_CERTIFICATE_NAME=$certificate_name"
)

if ! docker network inspect "$gateway_network" >/dev/null 2>&1; then
  echo "### Creating Docker network $gateway_network ..."
  docker network create "$gateway_network" >/dev/null
fi

mkdir -p "$study_repo_path/deploy/certbot/conf" \
  "$study_repo_path/deploy/certbot/www"

echo "### Starting the study services (without a second public nginx) ..."
"${study_compose[@]}" up -d --build v2 v3 dispatcher certbot

live_dir="$study_repo_path/deploy/certbot/conf/live/$certificate_name"
renewal_file="$study_repo_path/deploy/certbot/conf/renewal/$certificate_name.conf"
dummy_created=0
if [ ! -s "$live_dir/fullchain.pem" ] || [ ! -s "$live_dir/privkey.pem" ]; then
  echo "### Creating a temporary certificate so the shared nginx can start ..."
  mkdir -p "$live_dir"
  "${study_compose[@]}" run --rm --entrypoint openssl certbot \
    req -x509 -nodes -newkey rsa:2048 -days 1 \
    -keyout "/etc/letsencrypt/live/$certificate_name/privkey.pem" \
    -out "/etc/letsencrypt/live/$certificate_name/fullchain.pem" \
    -subj "/CN=$apex_domain"
  dummy_created=1
elif [ ! -s "$renewal_file" ]; then
  # A prior bootstrap may have been interrupted after nginx loaded the dummy
  # but before Certbot created its renewal lineage. Treat those existing PEM
  # files as temporary so a retry removes them before requesting the real cert.
  echo "### Reusing the temporary certificate from an interrupted bootstrap ..."
  dummy_created=1
fi

echo "### Starting/reloading the MaglioSite nginx as the shared gateway ..."
if ! env "${gateway_env[@]}" "${gateway_compose[@]}" up -d nginx; then
  echo >&2
  echo "Error: the shared gateway could not start. Current port owners:" >&2
  docker ps --filter publish=80 \
    --format '  port 80: {{.Names}} ({{.Image}}) {{.Ports}}' >&2 || true
  docker ps --filter publish=443 \
    --format '  port 443: {{.Names}} ({{.Image}}) {{.Ports}}' >&2 || true
  echo "Stop only the obsolete ingress container, then rerun this script." >&2
  exit 1
fi

restore_dummy_on_error() {
  if [ "$dummy_created" = "1" ]; then
    echo "Certificate request failed; restoring a temporary certificate for safe restarts." >&2
    mkdir -p "$live_dir"
    "${study_compose[@]}" run --rm --entrypoint openssl certbot \
      req -x509 -nodes -newkey rsa:2048 -days 1 \
      -keyout "/etc/letsencrypt/live/$certificate_name/privkey.pem" \
      -out "/etc/letsencrypt/live/$certificate_name/fullchain.pem" \
      -subj "/CN=$apex_domain" >/dev/null 2>&1 || true
  fi
}
trap restore_dummy_on_error ERR

if [ "$dummy_created" = "1" ]; then
  echo "### Removing the temporary certificate ..."
  "${study_compose[@]}" run --rm --entrypoint rm certbot \
    -rf "/etc/letsencrypt/live/$certificate_name" \
    "/etc/letsencrypt/archive/$certificate_name" \
    "/etc/letsencrypt/renewal/$certificate_name.conf"
fi

certbot_args=(
  certonly --webroot -w /var/www/certbot
  --cert-name "$certificate_name"
  --rsa-key-size 4096
  --agree-tos --no-eff-email --non-interactive --expand
)
for domain in "${domains[@]}"; do
  certbot_args+=(-d "$domain")
done
if [ "${certbot_staging:-0}" != "0" ]; then
  certbot_args+=(--staging)
fi
if [ "${FORCE_RENEWAL:-0}" != "0" ]; then
  certbot_args+=(--force-renewal)
fi
if [ -n "$certbot_email" ] && [[ "$certbot_email" != *example.com ]]; then
  certbot_args+=(--email "$certbot_email")
else
  certbot_args+=(--register-unsafely-without-email)
fi

echo "### Requesting one certificate for all study hostnames ..."
"${study_compose[@]}" run --rm --entrypoint certbot certbot "${certbot_args[@]}"
dummy_created=0
trap - ERR

echo "### Testing and reloading the shared nginx ..."
env "${gateway_env[@]}" "${gateway_compose[@]}" exec -T nginx nginx -t
env "${gateway_env[@]}" "${gateway_compose[@]}" exec -T nginx nginx -s reload

echo
echo "### Done. elisamaglio.de and the study now share the same nginx ingress."
