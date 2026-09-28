#!/bin/bash
# Canvas: download_qwen35_v1.sh
# Path: models_download.sh

set -e

# Resolve paths from the repository root, wherever the script is called from
cd "$(dirname "$0")"

echo "Starting offline model provisioning for the QA generation / Deep Research reasoner LLM..."

# Prefer the project venv so the model lands in the same environment Onyx runs in
PYTHON="venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"
HF="venv/bin/hf"
[ -x "$HF" ] || HF="hf"

# Mirror endpoint (override with HF_ENDPOINT=https://huggingface.co ./models_download.sh)
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Rust-accelerated transfers. huggingface_hub >= 1.0 replaced hf_transfer with hf_xet
# (HF_XET_HIGH_PERFORMANCE); HF_HUB_ENABLE_HF_TRANSFER is kept for older hub versions.
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DOWNLOAD_TIMEOUT=60

if ! "$PYTHON" -c "import huggingface_hub, hf_xet, hf_transfer" &> /dev/null; then
    echo "Installing huggingface_hub with hf_xet and hf_transfer..."
    "$PYTHON" -m pip install -U "huggingface_hub[hf_xet]" hf_transfer
fi

TARGET_DIR="models"
MODEL_FILE="Qwen3.5-9B-Q4_K_M.gguf"
mkdir -p "$TARGET_DIR"

echo "--------------------------------------------------------"
echo "Downloading Qwen3.5-9B Q4_K_M GGUF (Phase: QA Generation + Deep Research Reasoner) from $HF_ENDPOINT..."

# Only the text-model weights (~5.7 GB); the vision encoder (mmproj) is not used
"$HF" download unsloth/Qwen3.5-9B-GGUF "$MODEL_FILE" \
    --local-dir "$TARGET_DIR" \
    --max-workers 8

if [ ! -f "$TARGET_DIR/$MODEL_FILE" ]; then
    echo "❌ Download finished but $TARGET_DIR/$MODEL_FILE is missing."
    exit 1
fi

echo "--------------------------------------------------------"
echo "✅ Qwen3.5-9B successfully downloaded to $TARGET_DIR/$MODEL_FILE (QA_GENERATION_MODEL_PATH / QA_REASONER_MODEL_PATH)."
