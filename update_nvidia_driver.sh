#!/bin/bash
set -e

# Updates the NVIDIA driver via NVIDIA's own apt repo (Debian's own repo is
# stuck at 550.163.01, which only supports CUDA 12.4 - too old for the
# CUDA 13.x torch build already installed in venv/, which is why GPU tests
# fail with "The NVIDIA driver on your system is too old").
#
# WARNING:
#   - Requires sudo.
#   - This machine runs Hyprland/Wayland on NVIDIA - replacing the driver
#     will very likely kill your current graphical session immediately.
#   - A REBOOT is required afterward for the new driver to load.
#   - Save your work and close other apps before running this.
#
# After reboot, no other changes are needed: the venv already has
# torch/torchaudio/torchcodec built for CUDA 13, matched to each other -
# they'll just start working once the driver supports CUDA 13.

echo "=========================================================="
echo "WARNING: about to replace your NVIDIA driver. This will likely"
echo "crash your current graphical session and requires a reboot after."
echo "=========================================================="
read -p "Type 'yes' to continue: " CONFIRM
if [ "${CONFIRM}" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

WORK_DIR="$(mktemp -d)"
cd "${WORK_DIR}"

echo "--------------------------------------------------------"
echo "Adding NVIDIA's CUDA/driver apt repo..."
wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

cd /
rm -rf "${WORK_DIR}"

echo "--------------------------------------------------------"
echo "Removing the Debian-packaged NVIDIA driver to avoid conflicts..."
sudo apt-get remove --purge -y \
    nvidia-driver nvidia-driver-bin nvidia-driver-libs \
    nvidia-kernel-dkms nvidia-kernel-source \
    xserver-xorg-video-nvidia glx-alternative-nvidia 2>&1 || true

echo "--------------------------------------------------------"
echo "Installing latest NVIDIA driver from NVIDIA's own repo..."
sudo apt-get install -y cuda-drivers

echo "=========================================================="
echo "✅ Driver installed. You MUST reboot now for it to take effect:"
echo "     sudo reboot"
echo "After rebooting, verify with: nvidia-smi"
echo "(Should report CUDA Version 13.x, not 12.4.)"
echo "=========================================================="
