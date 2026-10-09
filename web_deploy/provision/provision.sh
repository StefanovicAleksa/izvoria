#!/bin/bash
# Run ONCE on the Vast GPU box, as root, after sync_to_box.sh. Idempotent: re-run it after a code sync.
#
# Vast template (docs/web_deploy.md): an nvidia/cuda *-cudnn-devel-ubuntu22.04 image (nvcc builds llama-server and
# llama-cpp-python), ONE 24 GB GPU (RTX 3090/4090), port 8095 exposed, on-start script
# `bash /workspace/onyx_v3/web_deploy/provision/onstart.sh`. Everything lives under /workspace: a stopped instance
# keeps its disk, so this runs once, not per start.
set -euo pipefail

ROOT=/workspace/onyx_v3
WEB=/workspace/web
ENV_FILE="${WEB}/api.env"
PGDATA=/workspace/pgdata
export DEBIAN_FRONTEND=noninteractive
cd "${ROOT}"

echo "== System packages"
apt-get update
apt-get install -y --no-install-recommends software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg lsb-release rsync ffmpeg git \
    build-essential cmake python3.11 python3.11-venv python3.11-dev supervisor openssl restic openssh-client tmux

echo "== PostgreSQL 15 + pgvector"
if [ ! -f /etc/apt/sources.list.d/pgdg.list ]; then
    install -d /usr/share/postgresql-common/pgdg
    curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc
    echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
        > /etc/apt/sources.list.d/pgdg.list
    apt-get update
fi
apt-get install -y --no-install-recommends postgresql-15 postgresql-15-pgvector
pg_dropcluster --stop 15 main 2>/dev/null || true

echo "== Secrets and settings"
install -d -m 700 "${WEB}"
if [ ! -f "${ENV_FILE}" ]; then
    umask 077
    sed -e "s/^WEB_API_SECRET=.*/WEB_API_SECRET=$(openssl rand -hex 32)/" \
        -e "s/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$(openssl rand -hex 24)/" \
        web_deploy/provision/api.env.example > "${ENV_FILE}"
fi
if grep -q '^RESTIC_PASSWORD=$' "${ENV_FILE}"; then
    sed -i "s/^RESTIC_PASSWORD=$/RESTIC_PASSWORD=$(openssl rand -hex 32)/" "${ENV_FILE}"
fi
set -a; . "${ENV_FILE}"; set +a

echo "== Backup key (the box logs in to the Storage Box with it)"
if [ ! -f "${WEB}/backup_ed25519" ]; then
    ssh-keygen -q -t ed25519 -N "" -C "onyx-gpu-box backup" -f "${WEB}/backup_ed25519"
fi

echo "== TLS for the API (the gateway pins this certificate; the Vast port is public)"
install -d -m 700 "${WEB}/tls"
if [ ! -f "${WEB}/tls/cert.pem" ]; then
    openssl req -x509 -newkey rsa:3072 -nodes -days 3650 -subj "/CN=onyx-gpu-box" \
        -keyout "${WEB}/tls/key.pem" -out "${WEB}/tls/cert.pem"
fi

echo "== Database cluster: the role, and the control database for the ingestion queue"
if [ ! -f "${PGDATA}/PG_VERSION" ]; then
    install -d -o postgres -g postgres "${PGDATA}"
    su postgres -c "/usr/lib/postgresql/15/bin/initdb -D ${PGDATA} --auth-local=peer --auth-host=scram-sha-256"
fi
su postgres -c "/usr/lib/postgresql/15/bin/pg_ctl -D ${PGDATA} -o '-c listen_addresses=127.0.0.1' -w start" || true
psql_admin() { su postgres -c "psql -v ON_ERROR_STOP=1 -tAq $*"; }
if [ -z "$(psql_admin "-c \"SELECT 1 FROM pg_roles WHERE rolname='${POSTGRES_USER}'\"")" ]; then
    # CREATEDB: the API creates each firm's database (onyx_firm_<slug>) at its first sign-in
    psql_admin "-c \"CREATE ROLE ${POSTGRES_USER} LOGIN CREATEDB PASSWORD '${POSTGRES_PASSWORD}'\""
fi
psql_admin "-c \"ALTER ROLE ${POSTGRES_USER} CREATEDB\""
if [ -z "$(psql_admin "-c \"SELECT 1 FROM pg_database WHERE datname='onyx_control'\"")" ]; then
    psql_admin "-c \"CREATE DATABASE onyx_control OWNER ${POSTGRES_USER}\""
fi
# pgvector is created per firm database by the API; a non-superuser may only do that for a trusted extension
psql_admin "-d template1 -c \"CREATE EXTENSION IF NOT EXISTS vector\""
su postgres -c "/usr/lib/postgresql/15/bin/pg_ctl -D ${PGDATA} -w stop"

echo "== Appliance venv, with the web API on top"
[ -x venv/bin/python ] || python3.11 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt -r web_deploy/api/requirements.txt
venv/bin/pip install --no-deps -r requirements-nodeps.txt
NVCC_VERSION="$(nvcc --version | grep -oP 'release \K[0-9.]+')"
if [ "$(cat venv/.llama_cpp_cuda_marker 2>/dev/null)" != "${NVCC_VERSION}" ]; then
    # Ingestion still runs its small GGUFs in-process
    CMAKE_ARGS="-DGGML_CUDA=on" venv/bin/pip install --force-reinstall --no-cache-dir llama-cpp-python
    echo "${NVCC_VERSION}" > venv/.llama_cpp_cuda_marker
fi

echo "== llama-server"
bash web_deploy/provision/build_llama_server.sh

echo "== supervisord"
cp web_deploy/provision/supervisord.conf /etc/supervisor/conf.d/onyx-web.conf
rm -f /etc/supervisor/conf.d/onyx.conf  # the noVNC demo's programs, if this box ran it before
install -d /workspace/tenants /workspace/source-cache /workspace/logs /workspace/samples

cat <<EOF

Provisioned. On the gateway VPS:
  1. Put this in /etc/onyx-web-gateway.env:   WEB_API_SECRET=${WEB_API_SECRET}
  2. Copy the API's certificate to the VPS:    ${WEB}/tls/cert.pem  ->  /etc/onyx-web/box-cert.pem
Backups (docs/web_deploy.md, "Backups"):
  3. Add this key to the Storage Box (Hetzner console, or ssh-copy-id -p 23 -s <user>@<user>.your-storagebox.de):
       $(cat "${WEB}/backup_ed25519.pub")
  4. Set RESTIC_REPOSITORY in ${ENV_FILE} if it is empty, and SAVE RESTIC_PASSWORD somewhere off this box.
     First backup, to check it all works:  venv/bin/python -m web_deploy.backup.runner --now
Start everything now with:  bash ${ROOT}/web_deploy/provision/onstart.sh
EOF
