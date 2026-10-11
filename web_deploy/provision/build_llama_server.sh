#!/bin/bash
# Builds llama.cpp's `llama-server` with CUDA, into /workspace/llama.cpp/build/bin/llama-server.
#
# llama-server is what lets several users be answered at once from ONE copy of Qwen3.5-9B: parallel slots that share
# the weights and (with --kv-unified) one KV cache. It reads the same GGUF the appliance uses.
#
# Pinned to the release that was validated: b11396 (2e7c58c, 2026-10-04) runs Qwen3.5's hybrid Gated DeltaNet layers
# with --parallel --kv-unified, and is what the dev spike and the local web version were tested on
# (docs/web_deploy.md, "M0 spike"). Never build master: a newer server can change what the client relies on (the
# raw JSON grammar it sends instead of response_format, which this release fails on for Qwen3.5). To move to a newer
# release, re-run the spike's checks on it, then change the pin here and in the docs:
#   LLAMA_CPP_REF=b11500 bash web_deploy/provision/build_llama_server.sh   # try one without changing the pin
set -euo pipefail

REF="${LLAMA_CPP_REF:-b11396}"
# The dev launcher builds into the repo's onyx_data/ instead (web_deploy/local/run_local.py)
SRC="${LLAMA_CPP_DIR:-/workspace/llama.cpp}"
command -v nvcc >/dev/null || export PATH="/usr/local/cuda/bin:${PATH}"

if [ ! -d "${SRC}/.git" ]; then
    git clone https://github.com/ggml-org/llama.cpp "${SRC}"
fi
git -C "${SRC}" fetch --tags origin
# --force: the patches below are applied on a clean tree every time, never on top of an older copy of themselves
git -C "${SRC}" checkout --force --detach "${REF}" 2>/dev/null || git -C "${SRC}" checkout --force --detach "origin/${REF}"

# Izvoria's own patches (web_deploy/provision/llama_patches/*.patch, in order), written against the pinned release.
# 0001: answers before long prompts: while answers are being written, a prompt longer than ONYX_SHORT_PROMPT_TOKENS
#       waits (and pending prompts are read shortest-first), so one long prompt cannot slow every chat to a crawl.
# Moving the pin means re-checking that each still applies and still does what it says (the spike's stress stage).
PATCHES="$(cd "$(dirname "$0")" && pwd)/llama_patches"
for patch in "${PATCHES}"/*.patch; do
    [ -e "${patch}" ] || continue
    git -C "${SRC}" apply --check "${patch}" || { echo "Patch $(basename "${patch}") does not apply to ${REF}." >&2; exit 1; }
    git -C "${SRC}" apply "${patch}"
    echo "Applied $(basename "${patch}")."
done
BUILD_ID="$(git -C "${SRC}" rev-parse HEAD)+$(cat "${PATCHES}"/*.patch 2>/dev/null | sha256sum | cut -c1-16)"

if [ "$(cat "${SRC}/.built_ref" 2>/dev/null)" = "${BUILD_ID}" ] && [ -x "${SRC}/build/bin/llama-server" ]; then
    echo "llama-server is already built at ${BUILD_ID}."
    exit 0
fi

# Static: one self-contained binary, so the build can be moved or the tree cleaned without breaking it
cmake -S "${SRC}" -B "${SRC}/build" -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "${SRC}/build" --target llama-server -j "$(nproc)"
echo "${BUILD_ID}" > "${SRC}/.built_ref"
echo "Built $("${SRC}/build/bin/llama-server" --version 2>&1 | head -1)"
