#!/bin/bash
# Run ONCE on the Vast GPU box, as root, after sync_to_box.sh. Idempotent: re-run it after a code sync.
#
# Vast template (docs/web_deploy.md): an nvidia/cuda *-cudnn-devel-ubuntu22.04 image (nvcc builds llama-server and
# llama-cpp-python), ONE 24 GB GPU (RTX 3090/4090), NO ports exposed (the box only dials out), on-start script
# `bash /workspace/onyx_v3/web_deploy/provision/onstart.sh`. Code, models and settings live under /workspace (a
# stopped instance keeps its disk, so this runs once, not per start); client data never does: the databases and
# vaults are on the Hetzner VPS, reached through the tunnel (docs/web_deploy.md, "Where data lives").
#
# Needs VPS_HOST (the VPS's address) in api.env, or in the environment the first time:
#   VPS_HOST=vps.example.com bash /workspace/onyx_v3/web_deploy/provision/provision.sh
set -euo pipefail

ROOT=/workspace/onyx_v3
WEB=/workspace/web
ENV_FILE="${WEB}/api.env"
export DEBIAN_FRONTEND=noninteractive
# Keep a config file the image already changed (Vast's base image ships its own supervisord.conf) instead of
# stopping at dpkg's "keep or replace?" question, which aborts the install when nobody can answer it
APT_KEEP_CONF=(-o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold)
cd "${ROOT}"
# Finish anything an earlier, interrupted run left half-configured
dpkg --force-confdef --force-confold --configure -a || true

echo "== System packages"
apt-get update
apt-get install -y "${APT_KEEP_CONF[@]}" --no-install-recommends software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y "${APT_KEEP_CONF[@]}" --no-install-recommends ca-certificates curl gnupg lsb-release rsync ffmpeg git \
    build-essential cmake python3.11 python3.11-venv python3.11-dev supervisor openssl openssh-client tmux \
    libpq-dev
# A box provisioned before the data moved to the VPS ran its own Postgres; nothing here may keep a database
if [ -d /workspace/pgdata ]; then
    echo "!! /workspace/pgdata is a database from before the data moved to the VPS. Back it up and restore it on the"
    echo "!! VPS (docs/web_deploy.md, \"Moving existing data\"), then delete it: rm -rf /workspace/pgdata"
fi

echo "== Secrets and settings"
install -d -m 700 "${WEB}"
if [ ! -f "${ENV_FILE}" ]; then
    umask 077
    # The secret and the database password are the VPS's (install_gateway.sh makes them): copy them in after
    sed -e "s/^VPS_HOST=.*/VPS_HOST=${VPS_HOST:-}/" web_deploy/provision/api.env.example > "${ENV_FILE}"
fi
set -a; . "${ENV_FILE}"; set +a
# api.env points TMPDIR at /workspace/tmp, which onstart.sh only makes at a start: the compilers below need it now
install -d -m 700 "${TMPDIR:-/workspace/tmp}"
: "${VPS_HOST:?set VPS_HOST in ${ENV_FILE} (the Hetzner VPS address), then run this again}"

echo "== The box's key for the VPS: its tunnels only (no shell, no files there)"
if [ ! -f "${WEB}/tunnel_ed25519" ]; then
    ssh-keygen -q -t ed25519 -N "" -C "izvoria-box tunnel" -f "${WEB}/tunnel_ed25519"
fi
# The file-staging key of earlier versions reached every firm's files on the VPS: no box has one any more
rm -f "${WEB}/sync_ed25519" "${WEB}/sync_ed25519.pub"

echo "== No database password on this box: it gets one from the gateway at each start, kept in memory only"
if grep -q '^POSTGRES_PASSWORD=..*' "${ENV_FILE}"; then
    sed -i 's/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=/' "${ENV_FILE}"
    echo "   removed the database password an earlier version kept in ${ENV_FILE}"
fi
for leftover in "${ROOT}/.env" "${ROOT}/.env.prod" "${ROOT}/.env.test"; do
    if [ -f "${leftover}" ] && grep -q '^POSTGRES_PASSWORD=..*' "${leftover}"; then
        echo "!! ${leftover} holds a database password; remove it (the box must keep none): rm ${leftover}" >&2
        exit 1
    fi
done
# rsync, for the VPS to move one job's files to and from the scratch through the box's own sshd
command -v rsync >/dev/null || apt-get install -y --no-install-recommends rsync

echo "== The VPS's host key, pinned (check the fingerprint against the VPS's own: ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub)"
if [ ! -s "${WEB}/known_hosts" ]; then
    ssh-keyscan -t ed25519 -p "${VPS_SSH_PORT:-22}" "${VPS_HOST}" > "${WEB}/known_hosts"
fi
ssh-keygen -lf "${WEB}/known_hosts"

echo "== Appliance venv, with the web API on top"
[ -x venv/bin/python ] || python3.11 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt -r web_deploy/api/requirements.txt
venv/bin/pip install --no-deps -r requirements-nodeps.txt
# PyTorch and its companions as CUDA 12.6 builds, the same versions pip just chose. PyPI's default builds target the
# newest CUDA (13.x) and refuse a host whose driver tops out lower ("driver too old"), and torchcodec's CUDA-13 build
# can't even load beside a 12.x torch. 12.6 runs on any host whose driver supports CUDA 12.6 or newer (Vast's
# "Max CUDA" column), which is nearly every rented GPU.
TORCH_CUDA="${TORCH_CUDA:-cu126}"
read -r TORCH_V TORCHAUDIO_V TORCHVISION_V TORCHCODEC_V < <(venv/bin/python - <<'PY'
import importlib.metadata as m
print(*(m.version(p).split("+")[0] for p in ("torch", "torchaudio", "torchvision", "torchcodec")))
PY
)
venv/bin/pip install --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
    "torch==${TORCH_V}+${TORCH_CUDA}" "torchaudio==${TORCHAUDIO_V}+${TORCH_CUDA}" "torchvision==${TORCHVISION_V}+${TORCH_CUDA}"
venv/bin/pip install --no-deps --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" "torchcodec==${TORCHCODEC_V}+${TORCH_CUDA}"
venv/bin/python -c "import torch; from torchcodec.decoders import AudioDecoder; assert torch.cuda.is_available(), 'torch sees no GPU'; print('torch', torch.__version__, 'on', torch.cuda.get_device_name(0))"
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
# Scratch for one job at a time (staging.py), at the same absolute path as the VPS's tenants; wiped after each job.
# Old keys from before the move (backup key, API certificate) are no longer used.
install -d /workspace/tenants /workspace/staging-state /workspace/logs /workspace/samples
rm -rf /workspace/source-cache /workspace/backup-staging

cat <<EOF

Provisioned. On the VPS, as root (docs/web_deploy.md, "Setting it up"):
  1. Let this box in (only its tunnels, and only while the gateway has the box started):
       izvoria-add-box-keys '$(cat "${WEB}/tunnel_ed25519.pub")'
  2. Copy into ${ENV_FILE} the value from /etc/onyx-web-gateway.env:  WEB_API_SECRET
     (no database password: the gateway gives this box a new one at every start)
Then start everything with:  bash ${ROOT}/web_deploy/provision/onstart.sh
EOF
