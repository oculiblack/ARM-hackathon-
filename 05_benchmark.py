#!/usr/bin/env python3
"""Raspberry Pi 5 telemetry & benchmark harness (SoC temperature, throttling, RAM, power)."""
import argparse
import csv
import importlib
import json
import os
import subprocess
import time
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

_cfg = importlib.import_module("01_config")
BENCHMARK = _cfg.BENCHMARK
HARDWARE = _cfg.HARDWARE


def read_temp_c() -> float:
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"], timeout=5).decode()
        return float(out.split("=")[1].split("'")[0])
    except Exception:
        return -1.0


def read_throttled() -> str:
    try:
        return subprocess.check_output(["vcgencmd", "get_throttled"], timeout=5).decode().strip()
    except Exception:
        return "unknown"


def read_ram_used_mb() -> float:
    try:
        for line in subprocess.check_output(["free", "-m"], timeout=5).decode().splitlines():
            if line.startswith("Mem:"):
                return float(line.split()[2])
    except Exception:
        return -1.0
    return -1.0


def read_power_watts() -> float:
    port = HARDWARE.get("power_meter_port", "") or os.environ.get("POWER_METER_PORT", "")
    if not port or not Path(port).exists():
        return -1.0
    try:
        import serial
        with serial.Serial(port, 9600, timeout=1) as ser:
            return float(ser.readline().decode().strip())
    except Exception:
        return -1.0


class Benchmark:
    def __init__(self, output_csv: str, duration_s: int = 300, interval_s: float = 1.0):
        self.output_csv = Path(output_csv)
        self.duration_s = duration_s
        self.interval_s = interval_s
        self.rows = []

    def sample(self, detector_latency_ms=None, llm_latency_ms=None) -> dict:
        row = {
            "ts": time.time(),
            "temp_c": read_temp_c(),
            "throttled": read_throttled(),
            "ram_used_mb": read_ram_used_mb(),
            "power_watts": read_power_watts(),
            "detector_latency_ms": detector_latency_ms if detector_latency_ms is not None else -1,
            "llm_latency_ms": llm_latency_ms if llm_latency_ms is not None else -1,
        }
        self.rows.append(row)
        return row

    def save(self):
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            return
        with open(self.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.rows[0].keys())
            w.writeheader()
            w.writerows(self.rows)
        print(f"[Benchmark] Saved {len(self.rows)} samples -> {self.output_csv}")

    def summary(self) -> dict:
        if not self.rows:
            return {}
        pct = lambda vals, p: round(float(np.percentile(vals, p)), 2) if (np is not None and vals) else None
        temps = [r["temp_c"] for r in self.rows if r["temp_c"] > 0]
        rams = [r["ram_used_mb"] for r in self.rows if r["ram_used_mb"] > 0]
        powers = [r["power_watts"] for r in self.rows if r["power_watts"] > 0]
        return {
            "samples": len(self.rows),
            "temp_p50_c": pct(temps, 50),
            "temp_max_c": round(max(temps), 1) if temps else None,
            "throttle_events": sum(1 for r in self.rows if "0x0" not in r["throttled"]),
            "ram_p50_mb": pct(rams, 50),
            "ram_peak_mb": round(max(rams), 1) if rams else None,
            "power_p50_w": pct(powers, 50),
        }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PS5 Pi 5 Telemetry & Benchmark Harness")
    p.add_argument("--test-sensors", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--duration", type=int, default=None)
    args = p.parse_args()
    if args.test_sensors:
        print("[TEST] Temp    :", read_temp_c(), "C")
        print("[TEST] Throttle:", read_throttled())
        print("[TEST] RAM     :", read_ram_used_mb(), "MB")
        print("[TEST] Power   :", read_power_watts(), "W")
    elif args.run:
        b = Benchmark(BENCHMARK["output_csv"], args.duration or BENCHMARK["duration_seconds"])
        t_end = time.time() + b.duration_s
        try:
            while time.time() < t_end:
                b.sample()
                time.sleep(b.interval_s)
        except KeyboardInterrupt:
            pass
        b.save()
        print("[Benchmark] Summary:", json.dumps(b.summary(), indent=2))
    else:
        p.print_help()
