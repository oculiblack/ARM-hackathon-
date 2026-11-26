#!/usr/bin/env python3
"""Append-only on-device inspection logger (CSV + SQLite, zero cloud dependence)."""
import argparse
import csv
import importlib
import sqlite3
import time
from pathlib import Path

_cfg = importlib.import_module("01_config")
LOGGING = _cfg.LOGGING


class InspectionLogger:
    def __init__(self, csv_path=None, db_path=None):
        self.csv_path = Path(csv_path or LOGGING["csv_path"])
        self.db_path = Path(db_path or LOGGING["db_path"])
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "ts", "frame", "defect_type", "severity", "action", "source",
                    "det_latency_ms", "llm_latency_ms", "ram_mb", "temp_c", "watts",
                ])
        con = sqlite3.connect(str(self.db_path))
        con.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, frame INTEGER, defect_type TEXT, severity TEXT,
                action TEXT, source TEXT, det_latency_ms REAL, llm_latency_ms REAL,
                ram_mb REAL, temp_c REAL, watts REAL
            )
        """)
        con.commit()
        con.close()

    def log(self, decision: dict, hw: dict | None = None):
        hw = hw or {}
        row = [
            decision.get("ts", time.time()), decision.get("frame", -1),
            decision.get("defect_type", "NR"), decision.get("severity", "NR"),
            decision.get("action", "NR"), decision.get("source", "NR"),
            decision.get("detector_latency_ms", -1), decision.get("llm_latency_ms", -1),
            hw.get("ram_mb", -1), hw.get("temp_c", -1), hw.get("watts", -1),
        ]
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        con = sqlite3.connect(str(self.db_path))
        con.execute(
            """INSERT INTO inspections
               (ts, frame, defect_type, severity, action, source,
                det_latency_ms, llm_latency_ms, ram_mb, temp_c, watts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            row,
        )
        con.commit()
        con.close()

    def read_recent(self, limit: int = 10) -> list:
        con = sqlite3.connect(str(self.db_path))
        rows = con.execute("SELECT * FROM inspections ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        con.close()
        return rows

    def count(self) -> int:
        con = sqlite3.connect(str(self.db_path))
        n = con.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
        con.close()
        return n


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PS5 Local Inspection Logger")
    p.add_argument("--test-log", action="store_true")
    p.add_argument("--read", action="store_true")
    args = p.parse_args()
    logger = InspectionLogger()
    if args.test_log:
        logger.log(
            {"ts": time.time(), "frame": 1, "defect_type": "crazing", "severity": "high",
             "action": "stop_line", "source": "template", "detector_latency_ms": 18.6, "llm_latency_ms": 0.0},
            {"ram_mb": 583.0, "temp_c": 47.7, "watts": -1},
        )
        print("[TEST] Total rows:", logger.count())
        print("[TEST] Latest row:", logger.read_recent(1)[0])
    elif args.read:
        for r in logger.read_recent(10):
            print(r)
    else:
        p.print_help()
