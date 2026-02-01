#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

is_wsl() {
  grep -qi microsoft /proc/version 2>/dev/null
}

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but not found. Install Python 3.11+ and rerun." >&2
  exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
  echo "sudo is required but not found. Install sudo and rerun." >&2
  exit 1
fi

if ! is_wsl && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA driver not detected. Installing recommended driver." >&2
  sudo apt-get update
  sudo apt-get install -y ubuntu-drivers-common
  sudo ubuntu-drivers install
  echo "Driver installation completed. A reboot is likely required." >&2
fi

CUDA_REPO_DEB="cuda-keyring_1.1-1_all.deb"
CUDA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/${CUDA_REPO_DEB}"

sudo apt-get update
sudo apt-get install -y wget ca-certificates gnupg

if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
  wget -q "${CUDA_REPO_URL}" -O "/tmp/${CUDA_REPO_DEB}"
  sudo dpkg -i "/tmp/${CUDA_REPO_DEB}"
fi

sudo apt-get update

CUDA_TOOLKIT_PKG=""
for candidate in cuda-toolkit-12-6 cuda-toolkit-12-5 cuda-toolkit-12-4 cuda-toolkit-12; do
  if apt-cache show "${candidate}" >/dev/null 2>&1; then
    CUDA_TOOLKIT_PKG="${candidate}"
    break
  fi
done

if [ -z "${CUDA_TOOLKIT_PKG}" ]; then
  echo "No supported cuda-toolkit package found in the NVIDIA repo." >&2
  exit 1
fi

sudo apt-get install -y "${CUDA_TOOLKIT_PKG}" libcudnn9-cuda-12 libcudnn9-dev-cuda-12

VENV_DIR="${REPO_ROOT}/.venv"
if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${REPO_ROOT}/requirements.txt"

if is_wsl; then
  echo "CUDA + cuDNN installed for WSL. If GPU is still not detected, restart WSL or your shell."
else
  echo "CUDA + cuDNN installed. Reboot if the NVIDIA driver was just installed."
fi
