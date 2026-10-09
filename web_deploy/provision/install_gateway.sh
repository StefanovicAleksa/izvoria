#!/bin/bash
# Run on the Hetzner VPS (Debian/Ubuntu), as root, from a copy of this repo at /opt/onyx_v3 with the web app built
# (web_deploy/gateway/static, made by `npm run build` in web_deploy/frontend on the dev box):
#   sudo bash /opt/onyx_v3/web_deploy/provision/install_gateway.sh onyx.example.com
# The VPS needs no GPU, models or appliance dependencies: only web_deploy/ and deploy/gateway/vast_client.py run here.
set -euo pipefail

DOMAIN="${1:?usage: install_gateway.sh <domain>}"
ROOT=/opt/onyx_v3
VENV=/opt/onyx-web-gateway

[ -f "${ROOT}/web_deploy/gateway/static/index.html" ] || {
    echo "Build the web app first (cd web_deploy/frontend && npm ci && npm run build) and copy it here." >&2; exit 1; }

apt-get update
apt-get install -y --no-install-recommends python3 python3-venv debian-keyring debian-archive-keyring \
    apt-transport-https curl gnupg openssl restic openssh-client

if ! command -v caddy >/dev/null; then
    curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt > /etc/apt/sources.list.d/caddy-stable.list
    apt-get update
    apt-get install -y caddy
fi

id onyx >/dev/null 2>&1 || useradd --system --home /nonexistent --shell /usr/sbin/nologin onyx
install -d -o onyx -g onyx -m 750 /var/lib/onyx-web
install -d -m 755 /etc/onyx-web
[ -x "${VENV}/bin/python" ] || python3 -m venv "${VENV}"
"${VENV}/bin/pip" install --upgrade pip
"${VENV}/bin/pip" install -r "${ROOT}/web_deploy/gateway/requirements.txt"

if [ ! -f /etc/onyx-web-gateway.env ]; then
    sed "s|^PUBLIC_URL=.*|PUBLIC_URL=https://${DOMAIN}|" "${ROOT}/web_deploy/provision/gateway.env.example" \
        > /etc/onyx-web-gateway.env
    chmod 600 /etc/onyx-web-gateway.env
    echo "Fill in /etc/onyx-web-gateway.env (WEB_API_SECRET, VAST_*, BOX_CERT_FILE, SMTP_*), then: systemctl restart onyx-web-gateway"
fi

# Backups of the gateway database: its own key for the Storage Box, readable only by the service user
if [ ! -f /etc/onyx-web/backup_ed25519 ]; then
    ssh-keygen -q -t ed25519 -N "" -C "onyx-gateway backup" -f /etc/onyx-web/backup_ed25519
fi
chown onyx:onyx /etc/onyx-web/backup_ed25519 /etc/onyx-web/backup_ed25519.pub
chmod 600 /etc/onyx-web/backup_ed25519
install -d -o onyx -g onyx -m 700 /var/lib/onyx-web/backup
install -m 644 "${ROOT}/web_deploy/provision/onyx-web-gateway-backup.service" /etc/systemd/system/
install -m 644 "${ROOT}/web_deploy/provision/onyx-web-gateway-backup.timer" /etc/systemd/system/

sed "s/onyx\.example\.com/${DOMAIN}/g" "${ROOT}/web_deploy/provision/Caddyfile" > /etc/caddy/Caddyfile
install -m 644 "${ROOT}/web_deploy/provision/onyx-web-gateway.service" /etc/systemd/system/onyx-web-gateway.service
systemctl daemon-reload
systemctl enable --now onyx-web-gateway
systemctl enable --now onyx-web-gateway-backup.timer
systemctl reload caddy

cat <<MSG
Gateway installed. Next:
  1. DNS: an A record for ${DOMAIN} -> this VPS
  2. Your owner account and the console's firm password:
       cd ${ROOT} && sudo -u onyx GATEWAY_DB=/var/lib/onyx-web/gateway.sqlite3 ${VENV}/bin/python -m web_deploy.gateway.admin_cli create-owner you@example.com
  3. Sign in at https://${DOMAIN} as the firm "admin", then as yourself, and create firms and their administrators
  4. Backups: add this key to the Storage Box, and set RESTIC_REPOSITORY/RESTIC_PASSWORD in /etc/onyx-web-gateway.env
     (the same as the GPU box's):
       $(cat /etc/onyx-web/backup_ed25519.pub)
     Check it:  systemctl start onyx-web-gateway-backup && journalctl -u onyx-web-gateway-backup -n 20
MSG
