#!/bin/bash
# Run on the DEV box: sets up (or updates) the Hetzner VPS that holds the firms' data and runs the gateway, in one go.
#
#   web_deploy/provision/setup_gateway.sh <server-ip> <domain>            # e.g. 2.31.10.175 app.izvoria.com
#   web_deploy/provision/setup_gateway.sh <server-ip> <domain> --no-install   # mount, build, copy only
#
# 1. checks the SSH login (root, your key)
# 2. mounts the Hetzner Volume at /workspace (formats it ONLY if it has no filesystem at all; never touches data)
# 3. checks that <domain> points at <server-ip> (HTTPS needs it)
# 4. builds the web app and copies the code to /opt/onyx_v3
# 5. runs install_gateway.sh <domain> (Postgres, the gateway, Caddy, the box's SSH account, backups)
# 6. prints a status report
#
# Safe to run again: every step checks before it changes anything. Secrets the installer prints are masked here; they
# stay in /etc/onyx-web-gateway.env on the server.
set -euo pipefail

IP="${1:?usage: setup_gateway.sh <server-ip> <domain> [--no-install]}"
DOMAIN="${2:?usage: setup_gateway.sh <server-ip> <domain> [--no-install]}"
INSTALL=1
[ "${3:-}" = "--no-install" ] && INSTALL=0

cd "$(dirname "$0")/../.."
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "root@${IP}")
step() { printf '\n== %s\n' "$*"; }
mask() { sed -E 's/(WEB_API_SECRET=|POSTGRES_PASSWORD=|RESTIC_PASSWORD=)[^[:space:]]+/\1<hidden: see \/etc\/onyx-web-gateway.env>/g'; }

step "1. Login"
"${SSH[@]}" 'echo "host: $(hostname)  os: $(lsb_release -ds 2>/dev/null)  cpus: $(nproc)  ram: $(free -h | awk "/Mem:/{print \$2}")"'

step "2. Volume at /workspace"
"${SSH[@]}" bash -s <<'REMOTE'
set -euo pipefail
DEV=$(ls /dev/disk/by-id/scsi-0HC_Volume_* 2>/dev/null | head -1 || true)
if [ -z "${DEV}" ]; then
    echo "!! no Hetzner Volume attached; /workspace stays on the system disk"
    mkdir -p /workspace
elif [ "$(findmnt -n -o SOURCE /workspace 2>/dev/null || true)" = "$(readlink -f "${DEV}")" ]; then
    echo "already mounted"
else
    if [ -z "$(blkid -o value -s TYPE "${DEV}" 2>/dev/null || true)" ]; then
        echo "the Volume has no filesystem yet: formatting it ext4"
        mkfs.ext4 -q -F "${DEV}"
    fi
    mkdir -p /workspace
    # Hetzner's automatic mount puts it at /mnt/HC_Volume_<id>: take it from there
    OLD=$(findmnt -n -o TARGET -S "$(readlink -f "${DEV}")" 2>/dev/null | head -1 || true)
    STAGE="${OLD}"
    if [ -z "${STAGE}" ]; then
        STAGE=$(mktemp -d)
        mount "${DEV}" "${STAGE}"
    fi
    # Anything already in /workspace (an install that ran before the Volume was mounted) moves onto it
    if [ -n "$(ls -A /workspace)" ]; then
        rsync -aHAX /workspace/ "${STAGE}/"
        echo "moved what /workspace held onto the Volume"
    fi
    umount "${STAGE}"
    [ "${STAGE}" != "${OLD}" ] && rmdir "${STAGE}"
    [ -n "${OLD}" ] && rmdir "${OLD}" 2>/dev/null || true
    sed -i "\|^${DEV} |d" /etc/fstab
    echo "${DEV} /workspace ext4 discard,nofail,defaults 0 0" >> /etc/fstab
    systemctl daemon-reload
    mount /workspace
    findmnt -n /workspace >/dev/null || { echo "!! the Volume did not mount at /workspace"; exit 1; }
    echo "mounted ${DEV} at /workspace"
    # Services started against the old directory pick up the Volume
    systemctl try-restart onyx-web-gateway onyx-web-backup 2>/dev/null || true
fi
df -h /workspace | tail -1

REMOTE

step "3. DNS: ${DOMAIN}"
RESOLVED=$(getent ahostsv4 "${DOMAIN}" 2>/dev/null | awk 'NR==1{print $1}' || true)
if [ "${RESOLVED}" = "${IP}" ]; then
    echo "ok: ${DOMAIN} -> ${IP}"
else
    echo "!! ${DOMAIN} resolves to '${RESOLVED:-nothing}', not ${IP}. Add an A record (host: ${DOMAIN%%.*}, value: ${IP})."
    echo "   Everything else still installs; HTTPS starts working once the record is there (Caddy keeps retrying)."
fi

step "4. Web app and code"
if [ ! -f web_deploy/gateway/static/index.html ] || [ -n "$(find web_deploy/frontend/src -newer web_deploy/gateway/static/index.html -print -quit)" ]; then
    (cd web_deploy/frontend && npm ci --no-audit --no-fund --silent && npm run build --silent) | tail -3
else
    echo "web app is up to date"
fi
"${SSH[@]}" 'mkdir -p /opt/onyx_v3'
rsync -az --delete \
    --exclude '/venv/' --exclude '/models/' --exclude '/onyx_data/' --exclude '/.git/' \
    --exclude 'node_modules/' --exclude '.venv/' --exclude '__pycache__/' --exclude '.pytest_cache/' \
    --exclude '/web_deploy/provision/prod/' --exclude '/web_deploy/local/local.env' --exclude '/.env' --exclude '/.env.*' \
    ./ "root@${IP}:/opt/onyx_v3/"
"${SSH[@]}" 'du -sh /opt/onyx_v3 | cut -f1 | sed "s/^/copied: /"'

if [ "${INSTALL}" = 1 ]; then
    step "5. install_gateway.sh ${DOMAIN} (a few minutes)"
    "${SSH[@]}" "bash /opt/onyx_v3/web_deploy/provision/install_gateway.sh ${DOMAIN}" 2>&1 | mask | grep -vE '^(Get|Hit|Ign|Reading|Building|Selecting|Preparing|Unpacking|Setting up|Processing|Created symlink|debconf|Fetched|  Downloading|Collecting|Using cached)' || {
        echo "!! the installer failed; the lines above say where"; exit 1; }
fi

step "6. Status"
"${SSH[@]}" DOMAIN="${DOMAIN}" bash -s <<'REMOTE'
for unit in postgresql@15-main onyx-web-gateway onyx-web-backup caddy ssh; do
    printf '  %-22s %s\n' "${unit}" "$(systemctl is-active "${unit}" 2>/dev/null || true)"
done
printf '  %-22s %s\n' "gateway answers" "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/auth/me 2>/dev/null | sed 's/^000$/no (see: journalctl -u onyx-web-gateway -n 20)/')"
printf '  %-22s %s\n' "box access" "$(cat /var/lib/onyx-web-box/access 2>/dev/null || echo 'not set up')"
printf '  %-22s %s\n' "box keys" "$(grep -s -c ssh-ed25519 /etc/onyx-web/box_authorized_keys || true) installed"
printf '  %-22s %s\n' "/workspace" "$(df -h /workspace | awk 'NR==2{print $4 " free of " $2}')"
printf '  %-22s %s\n' "settings to fill in" "$(grep -E '^(VAST_API_KEY|VAST_INSTANCE_ID|SMTP_HOST|RESTIC_REPOSITORY)=$' /etc/onyx-web-gateway.env 2>/dev/null | cut -d= -f1 | tr '\n' ' ')"
printf '  %-22s %s\n' "host key (for the box)" "$(ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub 2>/dev/null | awk '{print $2}')"
printf '  %-22s %s\n' "https://${DOMAIN}" "$(curl -s -o /dev/null -m 10 -w '%{http_code}' "https://${DOMAIN}/auth/me" 2>/dev/null | sed 's/^000$/not yet/')"
REMOTE
