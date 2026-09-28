#!/bin/bash
set -e

# Full environment setup for onyx_v3:
#   1. Build Python 3.11.9 from source in user space (matches the version this
#      project's venv originally used). No sudo required for this step.
#   2. Create venv/ with that interpreter and install requirements.txt.
#   3. Install the NVIDIA CUDA toolkit (nvcc) via apt so llama-cpp-python can
#      be built with GPU acceleration. Requires sudo - you'll be prompted for
#      your password on this step only.
#   4. Rebuild llama-cpp-python against CUDA.
#   5. Install Docker Engine + Compose plugin via apt. Requires sudo.
#   6. Start Docker and bring up the dev Postgres/pgvector container defined
#      in docker-compose.yaml (matches the credentials already in .env).
#
# Safe to re-run: steps that are already done (Python build, venv, toolkit,
# Docker, running container) are skipped, and pip/apt install commands are
# no-ops when already satisfied.

PYTHON_VERSION="3.11.9"
INSTALL_PREFIX="$HOME/.local/python-${PYTHON_VERSION}"
BUILD_DIR="/tmp/python-build-${PYTHON_VERSION}"
PYTHON_BIN="${INSTALL_PREFIX}/bin/python3.11"

cd "$(dirname "$0")"

# --- 1. Build Python 3.11.9 (user space, no sudo) ---
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "--------------------------------------------------------"
    echo "Downloading and building Python ${PYTHON_VERSION} from source..."
    rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
    curl -fsSL "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" -o "${BUILD_DIR}/Python.tgz"
    tar -xzf "${BUILD_DIR}/Python.tgz" -C "${BUILD_DIR}" --strip-components=1
    (
        cd "${BUILD_DIR}"
        ./configure --prefix="${INSTALL_PREFIX}" --with-ensurepip=install
        make -j"$(nproc)"
        make altinstall
    )
    rm -rf "${BUILD_DIR}"
else
    echo "Python ${PYTHON_VERSION} already built at ${INSTALL_PREFIX}, skipping build."
fi

# --- 2. venv + requirements.txt ---
if [ ! -x "venv/bin/python" ]; then
    echo "--------------------------------------------------------"
    echo "Creating virtual environment with Python ${PYTHON_VERSION}..."
    "${PYTHON_BIN}" -m venv venv
else
    echo "venv/ already exists, reusing it."
fi

echo "--------------------------------------------------------"
echo "Upgrading pip..."
venv/bin/python -m pip install --upgrade pip

echo "--------------------------------------------------------"
echo "Installing project dependencies from requirements.txt..."
venv/bin/pip install -r requirements.txt

# --- 3. CUDA toolkit / nvcc (needs sudo - you'll be prompted here) ---
# Prefers NVIDIA's own versioned packages (matches whatever driver/CUDA branch
# is currently installed, e.g. via update_nvidia_driver.sh) over Debian's
# nvidia-cuda-toolkit, which only ever tracks CUDA 12.4 and won't match a
# newer driver.
if ! command -v nvcc >/dev/null 2>&1; then
    echo "--------------------------------------------------------"
    echo "Installing CUDA compiler toolchain - sudo password required..."
    sudo apt update
    CUDA_BRANCH="$(apt-cache policy cuda-toolkit 2>/dev/null | awk '/Candidate:/ {print $2}' | grep -oP '^[0-9]+\.[0-9]+' | tr '.' '-')"
    if [ -n "${CUDA_BRANCH}" ]; then
        echo "Installing CUDA ${CUDA_BRANCH//-/.} compiler + dev libs (matches installed driver) via NVIDIA's repo..."
        sudo apt install -y "cuda-compiler-${CUDA_BRANCH}" "cuda-cudart-dev-${CUDA_BRANCH}" "libcublas-dev-${CUDA_BRANCH}"
    else
        echo "NVIDIA CUDA repo not found - falling back to Debian's nvidia-cuda-toolkit (CUDA 12.4)..."
        sudo apt install -y nvidia-cuda-toolkit
    fi
else
    echo "nvcc already installed, skipping CUDA toolkit install."
fi

# NVIDIA's versioned packages install under /usr/local/cuda-X.Y, not on PATH by default.
if ! command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(find /usr/local -maxdepth 1 -iname 'cuda-*' 2>/dev/null | sort -V | tail -1)"
    if [ -n "${CUDA_HOME}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
        export PATH="${CUDA_HOME}/bin:${PATH}"
        export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
        echo "Using nvcc from ${CUDA_HOME} for this run."
        echo "Add '${CUDA_HOME}/bin' to your shell's PATH permanently if you want 'nvcc' available in new terminals."
    fi
fi

# --- 4. Rebuild llama-cpp-python with CUDA support ---
# Marker lives inside venv/ so it resets automatically if the venv is ever recreated.
LLAMA_CUDA_MARKER="venv/.llama_cpp_cuda_marker"
if command -v nvcc >/dev/null 2>&1; then
    NVCC_VERSION="$(nvcc --version | grep -oP 'release \K[0-9.]+')"
    if [ -f "${LLAMA_CUDA_MARKER}" ] && [ "$(cat "${LLAMA_CUDA_MARKER}")" = "${NVCC_VERSION}" ]; then
        echo "llama-cpp-python already built with CUDA (nvcc ${NVCC_VERSION}), skipping rebuild."
    else
        echo "--------------------------------------------------------"
        echo "Rebuilding llama-cpp-python with CUDA acceleration..."
        CMAKE_ARGS="-DGGML_CUDA=on" venv/bin/pip install --force-reinstall --no-cache-dir llama-cpp-python
        echo "${NVCC_VERSION}" > "${LLAMA_CUDA_MARKER}"
    fi
else
    echo "nvcc not found - leaving llama-cpp-python as CPU-only build."
fi

# --- 5. Docker Engine + Compose plugin (needs sudo) ---
if ! command -v docker >/dev/null 2>&1; then
    echo "--------------------------------------------------------"
    echo "Installing Docker Engine - sudo password required..."
    sudo apt update
    sudo apt install -y ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
      $(. /etc/os-release && echo "${VERSION_CODENAME}") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    echo "Adding $(whoami) to the docker group (log out/in or run 'newgrp docker' for this to take effect without sudo)..."
    sudo usermod -aG docker "$(whoami)"
else
    echo "Docker already installed, skipping."
fi

echo "--------------------------------------------------------"
echo "Starting Docker service..."
sudo systemctl enable --now docker

# --- 6. Bring up the dev + test Postgres/pgvector containers ---
echo "--------------------------------------------------------"
echo "Starting databases (db_dev, db_test) via docker compose..."
sudo docker compose up -d db_dev db_test

echo "Waiting for Postgres (dev) to become ready..."
for i in $(seq 1 30); do
    if sudo docker exec onyx_postgres_dev pg_isready -U aleksa -d onyx_dev_db >/dev/null 2>&1; then
        echo "Dev Postgres is ready."
        break
    fi
    sleep 1
done

echo "Waiting for Postgres (test) to become ready..."
for i in $(seq 1 30); do
    if sudo docker exec onyx_postgres_test pg_isready -U postgres -d onyx_test_db >/dev/null 2>&1; then
        echo "Test Postgres is ready."
        break
    fi
    sleep 1
done

echo "--------------------------------------------------------"
echo "✅ Environment ready (Python ${PYTHON_VERSION}). Activate it with: source venv/bin/activate"
echo "   Dev database is up on localhost:5432 (db: onyx_dev_db)."
echo "   Test database is up on localhost:5433 (db: onyx_test_db) - needed for integration_tests."
echo "   Note: if this is Docker's first install, log out/in (or run 'newgrp docker')"
echo "   so you can run 'docker' commands without sudo."
