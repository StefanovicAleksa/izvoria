#!/bin/bash
# Run on the DEV box. Copies the code and the models Serbian/Slovenian need to the Vast instance.
#
#   deploy/instance/sync_to_box.sh root@203.0.113.9 41022          # code + models
#   deploy/instance/sync_to_box.sh root@203.0.113.9 41022 --code    # code only (after a git pull)
#
# The SSH host/port are on the instance's card in the Vast console ("Connect"). Re-runnable: rsync only sends changes.
set -euo pipefail

TARGET="${1:?usage: sync_to_box.sh user@host port [--code]}"
PORT="${2:?usage: sync_to_box.sh user@host port [--code]}"
MODE="${3:-all}"
REMOTE_ROOT="/workspace/onyx_v3"

cd "$(dirname "$0")/../.."
SSH="ssh -p ${PORT} -o StrictHostKeyChecking=accept-new"

echo "Syncing code to ${TARGET}:${REMOTE_ROOT} ..."
${SSH} "${TARGET}" "mkdir -p ${REMOTE_ROOT}/models"
rsync -az --delete -e "${SSH}" \
    --exclude '/venv/' --exclude '/deploy/.venv/' --exclude '/models/' --exclude '/onyx_data/' \
    --exclude '/input_video/' --exclude '/.env' --exclude '/.env.*' --exclude '/.git/' \
    --exclude '__pycache__/' --exclude '.pytest_cache/' --exclude '/test_output.txt' \
    ./ "${TARGET}:${REMOTE_ROOT}/"

[ "${MODE}" = "--code" ] && { echo "Code synced."; exit 0; }

# Every model path app/core/config.py and data/languages/{sr,sl}.toml reference (51 GB). Regenerate after adding a
# model:  grep -rhoE "[\"']models/[^\"']+[\"']" app data | sort -u
MODELS=(
    google_gemma-4-E4B-it-Q4_K_M.gguf
    llama-3.2-3b-instruct-q8_0.gguf
    Qwen3.5-9B-Q4_K_M.gguf
    Qwen2.5-7B-Instruct-Q4_K_M.gguf
    qwen2.5-3b-instruct-q4_k_m.gguf
    faster-whisper-large-v3
    whisper-tiny-en
    fullstop-punctuation-multilingual-base
    vad_multilingual_marblenet.nemo
    titanet_large.nemo
    qwen2-vl-7b-hf-bnb4bit
    Phi-3.5-mini-instruct
    flanec-base-cd
    deberta-v3-base-zeroshot
    bge-m3
    bge-reranker-v2-m3
    rapidocr
)
for model in "${MODELS[@]}"; do
    [ -e "models/${model}" ] || { echo "Missing models/${model} on this machine." >&2; exit 1; }
done

echo "Syncing ${#MODELS[@]} models (~51 GB on the first run) ..."
(cd models && rsync -az --partial --info=progress2 -e "${SSH}" --relative "${MODELS[@]}" "${TARGET}:${REMOTE_ROOT}/models/")
echo "Done. Next, on the box: bash ${REMOTE_ROOT}/deploy/instance/provision.sh"
