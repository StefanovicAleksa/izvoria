#!/bin/bash
# Run ONCE on the Vast instance, as root, after sync_to_box.sh. Idempotent: safe to re-run after a code sync.
#
# Expects the Vast template described in docs/deploy.md: an nvidia/cuda *-cudnn-devel-ubuntu22.04 image (nvcc is
# needed to build llama-cpp-python for CUDA), ports 8090 and 6080 exposed, and on-start script
# `bash /workspace/onyx_v3/deploy/instance/onstart.sh`.
#
# Everything lives under /workspace: a stopped Vast instance keeps its disk, so this runs once, not per start.
set -euo pipefail

ROOT=/workspace/onyx_v3
SECRETS=/workspace/secrets.env
PGDATA=/workspace/pgdata
AGENT_VENV=/workspace/agent-venv
export DEBIAN_FRONTEND=noninteractive

cd "${ROOT}"

echo "== System packages"
apt-get update
apt-get install -y --no-install-recommends software-properties-common
# Jammy's own python3.11 is a pre-release build; deadsnakes carries current 3.11, matching the dev venv (3.11.9).
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg lsb-release rsync ffmpeg git build-essential cmake \
    python3.11 python3.11-venv python3.11-dev \
    supervisor dbus-x11 xfce4 xfce4-terminal tigervnc-standalone-server novnc websockify \
    mpv fonts-dejavu fonts-noto-core libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-keysyms1 \
    libxcb-image0 libxcb-render-util0 libxcb-xinerama0 libgl1 libegl1

echo "== PostgreSQL 15 + pgvector (PGDG)"
if [ ! -f /etc/apt/sources.list.d/pgdg.list ]; then
    install -d /usr/share/postgresql-common/pgdg
    curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc
    echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" \
        > /etc/apt/sources.list.d/pgdg.list
    apt-get update
fi
apt-get install -y --no-install-recommends postgresql-15 postgresql-15-pgvector
# The packaged cluster would start its own postgres; supervisord runs ours from /workspace instead.
pg_dropcluster --stop 15 main 2>/dev/null || true

echo "== Secrets"
if [ ! -f "${SECRETS}" ]; then
    umask 077
    cat > "${SECRETS}" <<EOF
AGENT_SECRET=$(openssl rand -hex 32)
WEBSOCKIFY_TOKEN=$(openssl rand -hex 32)
POSTGRES_SERVER=127.0.0.1
POSTGRES_PORT=5432
POSTGRES_USER=onyx
POSTGRES_PASSWORD=$(openssl rand -hex 24)
ONYX_ROOT=${ROOT}
UPLOAD_ROOT=/workspace/uploads
EOF
fi
set -a; . "${SECRETS}"; set +a

echo "== Database cluster"
if [ ! -f "${PGDATA}/PG_VERSION" ]; then
    install -d -o postgres -g postgres "${PGDATA}"
    su postgres -c "/usr/lib/postgresql/15/bin/initdb -D ${PGDATA} --auth-local=peer --auth-host=scram-sha-256"
fi
su postgres -c "/usr/lib/postgresql/15/bin/pg_ctl -D ${PGDATA} -o '-c listen_addresses=127.0.0.1' -w start" || true
bash "${ROOT}/deploy/instance/create_databases.sh"
su postgres -c "/usr/lib/postgresql/15/bin/pg_ctl -D ${PGDATA} -w stop"

echo "== Appliance venv"
[ -x venv/bin/python ] || python3.11 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt
venv/bin/pip install --no-deps -r requirements-nodeps.txt
NVCC_VERSION="$(nvcc --version | grep -oP 'release \K[0-9.]+')"
if [ "$(cat venv/.llama_cpp_cuda_marker 2>/dev/null)" != "${NVCC_VERSION}" ]; then
    CMAKE_ARGS="-DGGML_CUDA=on" venv/bin/pip install --force-reinstall --no-cache-dir llama-cpp-python
    echo "${NVCC_VERSION}" > venv/.llama_cpp_cuda_marker
fi

echo "== Agent venv"
[ -x "${AGENT_VENV}/bin/python" ] || python3.11 -m venv "${AGENT_VENV}"
"${AGENT_VENV}/bin/pip" install -r deploy/instance/agent/requirements.txt

echo "== noVNC token (routes and authenticates the gateway's desktop proxy)"
install -d -m 700 /workspace/websockify
umask 077
echo "${WEBSOCKIFY_TOKEN}: 127.0.0.1:5901" > /workspace/websockify/tokens

echo "== supervisord"
cp deploy/instance/supervisord.conf /etc/supervisor/conf.d/onyx.conf
install -d /workspace/uploads /workspace/logs

cat <<EOF

Provisioned. Put these two values in the gateway's .env:
  AGENT_SECRET=${AGENT_SECRET}
  WEBSOCKIFY_TOKEN=${WEBSOCKIFY_TOKEN}

Start everything now with:  bash ${ROOT}/deploy/instance/onstart.sh
EOF
