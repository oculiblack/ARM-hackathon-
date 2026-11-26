#!/usr/bin/env bash
# =============================================================================
# 00_setup_pi5.sh — Raspberry Pi 5 one-time setup (CPU-Only, 64-bit Pi OS)
# =============================================================================
set -euo pipefail

echo "[1/5] Installing system packages..."
sudo apt-get update -y
sudo apt-get install -y \
    python3 python3-pip python3-venv python3-opencv \
    libopenblas-dev cmake ninja-build build-essential git curl v4l-utils bc

echo "[2/5] Creating Python virtual environment at ~/ps5venv2..."
python3 -m venv ~/ps5venv2
# shellcheck disable=SC1091
source ~/ps5venv2/bin/activate
pip install --upgrade pip setuptools wheel

echo "[3/5] Installing runtime Python dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "[4/5] Locking CPU governor to performance..."
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true

echo "[5/5] Creating project directories..."
mkdir -p models templates results logs
echo "Setup complete. Activate with: source ~/ps5venv2/bin/activate"
