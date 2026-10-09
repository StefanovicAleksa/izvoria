#!/bin/bash
# supervisord's llama-server: Qwen3.5-9B, resident for good, serving every Qwen role of every firm (answers, reasoned
# answers, report writing, retrieval scrubbing) from parallel slots.
#
# The defaults are the plan's starting point for one 24 GB card (docs/web_deploy.md, "VRAM"). The spike decides
# the final numbers; set them in /workspace/web/api.env, never by editing this file:
#   LLM_SERVER_PARALLEL   slots, i.e. requests decoded at once                       (4)
#   LLM_SERVER_KV_TOKENS  tokens the shared KV cache holds across all slots          (196608)
# --kv-unified lets one request use up to the whole cache, so a single report section or long conversation keeps its
# 128K context while short chats run beside it. The API admits requests against the same total (KvTokenBudget).
set -euo pipefail
set -a; . /workspace/web/api.env; set +a

exec /workspace/llama.cpp/build/bin/llama-server \
    --model "${ONYX_ROOT}/models/Qwen3.5-9B-Q4_K_M.gguf" \
    --host 127.0.0.1 --port 8081 \
    --n-gpu-layers 99 \
    --ctx-size "${LLM_SERVER_KV_TOKENS:-196608}" \
    --parallel "${LLM_SERVER_PARALLEL:-4}" \
    --kv-unified \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --flash-attn on \
    --jinja --reasoning-format none \
    --no-webui \
    --metrics
