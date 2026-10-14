#!/bin/bash
# Run on the Hetzner VPS (Debian 12 / Ubuntu 22.04+), as root, from a copy of this repo at /opt/onyx_v3 with the web
# app built (web_deploy/gateway/static, made by `npm run build` in web_deploy/frontend on the dev box):
#   sudo bash /opt/onyx_v3/web_deploy/provision/install_gateway.sh onyx.example.com
#
# The VPS holds every firm's data (docs/web_deploy.md, "Where data lives"): the Postgres databases, and the vaults and
# uploads under /workspace/tenants (mount a Hetzner Volume at /workspace first). The rented GPU box keeps none of it;
# it reaches the databases through an SSH tunnel it opens to this machine, as the user izvoria-box, whose key can
# only forward the tunnel's ports and is accepted only while the gateway has the box started; its database role works
# only then too. The VPS moves each job's files to and from the box itself; the box has no key to the files here. No GPU, models or appliance dependencies here: only web_deploy/ and deploy/'s small modules run.
set -euo pipefail

DOMAIN="${1:?usage: install_gateway.sh <domain>}"
ROOT=/opt/onyx_v3
VENV=/opt/onyx-web-gateway
ENV_FILE=/etc/onyx-web-gateway.env
DATA=/workspace
BOX_ACCESS_DIR=/var/lib/onyx-web-box

[ -f "${ROOT}/web_deploy/gateway/static/index.html" ] || {
    echo "Build the web app first (cd web_deploy/frontend && npm ci && npm run build) and copy it here." >&2; exit 1; }
mountpoint -q "${DATA}" || echo "!! ${DATA} is not a mount point. Attach a Hetzner Volume there for the firms' data."

apt-get update
apt-get install -y --no-install-recommends python3 python3-venv debian-keyring debian-archive-keyring \
    apt-transport-https curl gnupg lsb-release openssl restic openssh-client openssh-server rsync ffmpeg

if ! command -v caddy >/dev/null; then
    curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt > /etc/apt/sources.list.d/caddy-stable.list
    apt-get update
    apt-get install -y caddy
fi

echo "== Settings (generated once; keep a copy of the secrets off this machine)"
if [ ! -f "${ENV_FILE}" ]; then
    umask 077
    sed -e "s|^PUBLIC_URL=.*|PUBLIC_URL=https://${DOMAIN}|" \
        -e "s/^WEB_API_SECRET=.*/WEB_API_SECRET=$(openssl rand -hex 32)/" \
        -e "s/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$(openssl rand -hex 24)/" \
        -e "s/^RESTIC_PASSWORD=.*/RESTIC_PASSWORD=$(openssl rand -hex 32)/" \
        "${ROOT}/web_deploy/provision/gateway.env.example" > "${ENV_FILE}"
    chmod 600 "${ENV_FILE}"
    umask 022
fi
set -a; . "${ENV_FILE}"; set +a

echo "== PostgreSQL 15 + pgvector: every firm database and the ingestion queue, on loopback only"
if [ ! -f /etc/apt/sources.list.d/pgdg.list ]; then
    install -d /usr/share/postgresql-common/pgdg
    curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc
    echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
        > /etc/apt/sources.list.d/pgdg.list
    apt-get update
fi
apt-get install -y --no-install-recommends postgresql-15 postgresql-15-pgvector
# The box connects through its tunnel, which arrives here as a loopback connection: nothing listens beyond it
sed -i "s/^#\?listen_addresses.*/listen_addresses = 'localhost'/" /etc/postgresql/15/main/postgresql.conf
sed -i "s/^#\?max_connections.*/max_connections = 300/" /etc/postgresql/15/main/postgresql.conf
systemctl enable --now postgresql@15-main
systemctl restart postgresql@15-main
psql_admin() { su postgres -c "psql -v ON_ERROR_STOP=1 -tAq $*"; }
if [ -z "$(psql_admin "-c \"SELECT 1 FROM pg_roles WHERE rolname='${POSTGRES_USER}'\"")" ]; then
    # CREATEDB: the box's API creates each firm's database (onyx_firm_<slug>) at its first sign-in
    psql_admin "-c \"CREATE ROLE ${POSTGRES_USER} LOGIN CREATEDB PASSWORD '${POSTGRES_PASSWORD}'\""
fi
psql_admin "-c \"ALTER ROLE ${POSTGRES_USER} CREATEDB PASSWORD '${POSTGRES_PASSWORD}'\""
if [ -z "$(psql_admin "-c \"SELECT 1 FROM pg_database WHERE datname='onyx_control'\"")" ]; then
    psql_admin "-c \"CREATE DATABASE onyx_control OWNER ${POSTGRES_USER}\""
fi
# pgvector is created per firm database by the API; a non-superuser may only do that for a trusted extension
psql_admin "-d template1 -c \"CREATE EXTENSION IF NOT EXISTS vector\""
# The GPU box's own role. It creates and owns the firm databases (its API migrates them), and may log in only while the
# gateway has the box started, with a password changed at every start (izvoria-box-access). The gateway's role is a
# member, so the queue, backups and restores keep working on what the box owns.
if [ -z "$(psql_admin "-c \"SELECT 1 FROM pg_roles WHERE rolname='izvoria_box'\"")" ]; then
    psql_admin "-c \"CREATE ROLE izvoria_box NOLOGIN CREATEDB\""
fi
psql_admin "-c \"GRANT izvoria_box TO ${POSTGRES_USER}\""
psql_admin "-d onyx_control -c \"GRANT ALL ON SCHEMA public TO izvoria_box\""
psql_admin "-d onyx_control -c \"GRANT ALL ON ALL TABLES IN SCHEMA public TO izvoria_box\""
psql_admin "-d onyx_control -c \"ALTER DEFAULT PRIVILEGES FOR ROLE ${POSTGRES_USER} IN SCHEMA public GRANT ALL ON TABLES TO izvoria_box\""

echo "== Users and the data directories"
id onyx >/dev/null 2>&1 || useradd --system --home /nonexistent --shell /usr/sbin/nologin onyx
getent group izvoria-data >/dev/null || groupadd --system izvoria-data
# The GPU box's account: a shell only so sshd can run the forced commands below; no password, no key file of its own
id izvoria-box >/dev/null 2>&1 || useradd --system --create-home --home-dir /var/lib/izvoria-box --shell /bin/sh izvoria-box
usermod -aG izvoria-data onyx
usermod -aG izvoria-data izvoria-box
install -d -o onyx -g onyx -m 750 /var/lib/onyx-web
install -d -m 755 /etc/onyx-web
# Vaults and uploads: the gateway writes uploads, the box's rsync writes vaults, both read each other's (setgid group)
install -d -o root -g izvoria-data -m 2770 "${DATA}/tenants"
install -d -o onyx -g onyx -m 750 "${DATA}/source-cache"

echo "== The GPU box's SSH access (tunnel + file staging), open only while the gateway has the box started"
install -m 755 "${ROOT}/web_deploy/provision/izvoria-box-keys" /usr/local/bin/izvoria-box-keys
install -m 755 "${ROOT}/web_deploy/provision/add_box_keys.sh" /usr/local/sbin/izvoria-add-box-keys
[ -f /etc/onyx-web/box_authorized_keys ] || install -m 644 /dev/null /etc/onyx-web/box_authorized_keys
# The gateway (user onyx) says whether the box is started; sshd's key command (user nobody) reads it
install -d -o onyx -g onyx -m 755 "${BOX_ACCESS_DIR}"
[ -f "${BOX_ACCESS_DIR}/access" ] || { echo closed > "${BOX_ACCESS_DIR}/access"; chown onyx:onyx "${BOX_ACCESS_DIR}/access"; }
# Root applies every change of that file: the box's database role on or off, its sessions cut when off
install -m 755 "${ROOT}/web_deploy/provision/izvoria-box-access" /usr/local/sbin/izvoria-box-access
install -m 644 "${ROOT}/web_deploy/provision/izvoria-box-access.path" /etc/systemd/system/
install -m 644 "${ROOT}/web_deploy/provision/izvoria-box-access.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now izvoria-box-access.path
systemctl enable izvoria-box-access.service
systemctl start izvoria-box-access.service
# The VPS moves one job's files to and from the box itself (gateway/staging_server.py), with its own key, through the
# box's tunnel; the box has no key that reaches the files here
if [ ! -f /etc/onyx-web/staging_ed25519 ]; then
    ssh-keygen -q -t ed25519 -N "" -C "onyx-vps staging" -f /etc/onyx-web/staging_ed25519
fi
chown onyx:onyx /etc/onyx-web/staging_ed25519 /etc/onyx-web/staging_ed25519.pub
chmod 600 /etc/onyx-web/staging_ed25519
install -d -o onyx -g onyx -m 700 /var/lib/onyx-web/staging
cat > /etc/ssh/sshd_config.d/izvoria-box.conf <<SSHD
# Installed by install_gateway.sh. The GPU box logs in as izvoria-box with its tunnel key, confined in
# /etc/onyx-web/box_authorized_keys (port forwarding only: Postgres and the gateway here, its API and sshd back),
# and offered by izvoria-box-keys only while ${BOX_ACCESS_DIR}/access says "open".
Match User izvoria-box
    AuthorizedKeysFile none
    AuthorizedKeysCommand /usr/local/bin/izvoria-box-keys %u
    AuthorizedKeysCommandUser nobody
    PasswordAuthentication no
    KbdInteractiveAuthentication no
    PermitTTY no
    X11Forwarding no
    AllowAgentForwarding no
    AllowStreamLocalForwarding no
    GatewayPorts no
SSHD
sshd -t
systemctl reload ssh 2>/dev/null || systemctl reload sshd

echo "== The gateway"
[ -x "${VENV}/bin/python" ] || python3 -m venv "${VENV}"
"${VENV}/bin/pip" install --upgrade pip
"${VENV}/bin/pip" install -r "${ROOT}/web_deploy/gateway/requirements.txt"

# Backups (the firms' databases and vaults, and the gateway's own database): keys for the Storage Box, readable only
# by the service user
if [ ! -f /etc/onyx-web/backup_ed25519 ]; then
    ssh-keygen -q -t ed25519 -N "" -C "onyx-vps backup" -f /etc/onyx-web/backup_ed25519
fi
chown onyx:onyx /etc/onyx-web/backup_ed25519 /etc/onyx-web/backup_ed25519.pub
chmod 600 /etc/onyx-web/backup_ed25519
install -d -o onyx -g onyx -m 700 /var/lib/onyx-web/backup /var/lib/onyx-web/backup-staging
for unit in onyx-web-gateway.service onyx-web-gateway-backup.service onyx-web-gateway-backup.timer onyx-web-backup.service; do
    install -m 644 "${ROOT}/web_deploy/provision/${unit}" /etc/systemd/system/
done

sed "s/onyx\.example\.com/${DOMAIN}/g" "${ROOT}/web_deploy/provision/Caddyfile" > /etc/caddy/Caddyfile
systemctl daemon-reload
systemctl enable --now onyx-web-gateway
systemctl enable --now onyx-web-gateway-backup.timer
systemctl enable --now onyx-web-backup
systemctl restart onyx-web-gateway
systemctl reload caddy

cat <<MSG
Gateway installed. Next:
  1. DNS: an A record for ${DOMAIN} -> this VPS
  2. Fill in ${ENV_FILE}: VAST_API_KEY, VAST_INSTANCE_ID, SMTP_*; then systemctl restart onyx-web-gateway
  3. The GPU box needs, in its /workspace/web/api.env:
       VPS_HOST=<this VPS's address>   WEB_API_SECRET=${WEB_API_SECRET}   (no database password: it gets one per start)
     and this VPS needs the box's tunnel key, which its provision.sh prints:  izvoria-add-box-keys '<tunnel key>'
     The box pins this host key (compare with what provision.sh shows):
       $(ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub)
  4. Your owner account and the console's firm password:
       cd ${ROOT} && sudo -u onyx GATEWAY_DB=/var/lib/onyx-web/gateway.sqlite3 ${VENV}/bin/python -m web_deploy.gateway.admin_cli create-owner you@example.com
  5. Sign in at https://${DOMAIN} as the firm "admin", then as yourself, and create firms and their administrators
  6. Backups: add this key to the Storage Box, set RESTIC_REPOSITORY in ${ENV_FILE}, and SAVE RESTIC_PASSWORD off
     this machine (without it the backups cannot be read):
       $(cat /etc/onyx-web/backup_ed25519.pub)
     Check it (the first backup starts at once):  systemctl restart onyx-web-backup && journalctl -fu onyx-web-backup
MSG
