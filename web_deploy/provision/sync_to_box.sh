#!/bin/bash
# Run on the DEV box: copies the code (with the built web app) and the models to the Vast GPU box.
#
#   web_deploy/provision/sync_to_box.sh root@203.0.113.9 41022          # code + models
#   web_deploy/provision/sync_to_box.sh root@203.0.113.9 41022 --code    # code only
#   web_deploy/provision/sync_to_box.sh root@203.0.113.9 41022 --live    # code + the live models only (5.3 GB +
#                                                                          the embedder and reranker): enough for
#                                                                          the spike's sweep and stress stages
set -euo pipefail

TARGET="${1:?usage: sync_to_box.sh user@host port [--code]}"
PORT="${2:?usage: sync_to_box.sh user@host port [--code]}"
MODE="${3:-all}"
REMOTE_ROOT="/workspace/onyx_v3"

cd "$(dirname "$0")/../.."
SSH="ssh -p ${PORT} -o StrictHostKeyChecking=accept-new"

${SSH} "${TARGET}" "mkdir -p ${REMOTE_ROOT}/models"
rsync -az --delete -e "${SSH}" \
    --exclude '/venv/' --exclude '/deploy/.venv/' --exclude '/web_deploy/.venv/' \
    --exclude '/web_deploy/frontend/node_modules/' --exclude '/models/' --exclude '/onyx_data/' \
    --exclude '/input_video/' --exclude '/.env' --exclude '/.env.*' --exclude '/.git/' \
    --exclude '/web_deploy/provision/prod/' --exclude '/web_deploy/local/local.env' \
    --exclude '__pycache__/' --exclude '.pytest_cache/' \
    ./ "${TARGET}:${REMOTE_ROOT}/"
[ "${MODE}" = "--code" ] && { echo "Code synced."; exit 0; }

# The deploy/ list is the appliance's full model set; the hosted box needs the same files
MODELS=$(sed -n '/^MODELS=(/,/^)/p' deploy/instance/sync_to_box.sh | sed '1d;$d' | tr -d ' ')
# --live: what the live side serves (llama-server's 9B, the pinned embedder and reranker); no ingestion models
[ "${MODE}" = "--live" ] && MODELS="Qwen3.5-9B-Q4_K_M.gguf bge-m3 bge-reranker-v2-m3"
for model in ${MODELS}; do
    [ -e "models/${model}" ] || { echo "Missing models/${model} on this machine." >&2; exit 1; }
done
(cd models && rsync -az --partial --info=progress2 -e "${SSH}" --relative ${MODELS} "${TARGET}:${REMOTE_ROOT}/models/")
echo "Done. Next, on the box: bash ${REMOTE_ROOT}/web_deploy/provision/provision.sh"
