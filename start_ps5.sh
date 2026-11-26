#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source ~/ps5venv2/bin/activate
pkill -f "02_detector.py|04_gateway.py|07_dashboard.py" 2>/dev/null || true
rm -f results/events.jsonl /dev/shm/ps5_live.jpg
trap 'kill 0' EXIT INT TERM
python3 07_dashboard.py &
python3 04_gateway.py --run &
sleep 1
python3 02_detector.py --run --source 0
