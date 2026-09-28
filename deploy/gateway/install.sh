#!/bin/bash
# Run on the gateway VPS (Debian/Ubuntu), as root, from a copy of this repo at /opt/onyx_v3:
#   sudo bash /opt/onyx_v3/deploy/gateway/install.sh demo.example.com
# Only deploy/ is used here; the VPS needs no GPU, models or appliance dependencies.
set -euo pipefail

DOMAIN="${1:?usage: install.sh <domain>}"
ROOT=/opt/onyx_v3
VENV=/opt/onyx-gateway

apt-get update
apt-get install -y --no-install-recommends python3 python3-venv debian-keyring debian-archive-keyring \
    apt-transport-https curl gnupg

if ! command -v caddy >/dev/null; then
    curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt > /etc/apt/sources.list.d/caddy-stable.list
    apt-get update
    apt-get install -y caddy
fi

id onyx >/dev/null 2>&1 || useradd --system --home /nonexistent --shell /usr/sbin/nologin onyx
[ -x "${VENV}/bin/python" ] || python3 -m venv "${VENV}"
"${VENV}/bin/pip" install --upgrade pip
"${VENV}/bin/pip" install -r "${ROOT}/deploy/gateway/requirements.txt"

if [ ! -f /etc/onyx-gateway.env ]; then
    install -m 600 "${ROOT}/deploy/gateway/gateway.env.example" /etc/onyx-gateway.env
    sed -i "s/^SESSION_SECRET=.*/SESSION_SECRET=$(openssl rand -hex 32)/" /etc/onyx-gateway.env
    echo "Fill in /etc/onyx-gateway.env, then: systemctl restart onyx-gateway"
fi

sed "s/demo.example.com/${DOMAIN}/" "${ROOT}/deploy/gateway/Caddyfile" > /etc/caddy/Caddyfile
install -m 644 "${ROOT}/deploy/gateway/onyx-gateway.service" /etc/systemd/system/onyx-gateway.service
systemctl daemon-reload
systemctl enable --now onyx-gateway
systemctl reload caddy

echo "Gateway installed at https://${DOMAIN}"
