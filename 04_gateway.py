#!/usr/bin/env python3
"""Event-triggered tiered gateway: fast templates for known defects, ExecuTorch Llama for novel/uncertain cases."""
import argparse
import importlib
import json
import time
from pathlib import Path

_cfg = importlib.import_module("01_config")
_la = importlib.import_module("03_llama_advisor")
_bench = importlib.import_module("05_benchmark")
_log = importlib.import_module("06_logger")

GATING = _cfg.GATING
TEMPLATES_DIR = _cfg.TEMPLATES_DIR


def load_template(defect_type: str) -> dict | None:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in defect_type.lower())
    p = TEMPLATES_DIR / f"{safe}.json"
    return json.loads(p.read_text()) if p.exists() else None


def tiered_response(defect_type: str, confidence: float, bbox: list, advisor) -> dict:
    known = defect_type.lower() in [d.lower() for d in GATING["known_defects"]]
    ambiguous = confidence < GATING["confidence_threshold_llm"]
    tmpl = load_template(defect_type) if known else None

    if tmpl and not ambiguous:
        return {
            "source": "template",
            "defect_type": defect_type,
            "severity": tmpl["severity"],
            "recommendation": tmpl["recommendation"],
            "confidence": round(confidence, 2),
            "action": tmpl["action"],
            "llm_woken": False,
            "llm_latency_ms": 0.0,
            "bbox": bbox,
        }

    advice = advisor.advise(defect_type=defect_type, confidence=confidence, bbox=bbox)
    advice["source"] = "llm"
    advice["llm_woken"] = True
    advice["bbox"] = bbox
    return advice


def run_gateway():
    advisor = _la.LlamaAdvisor(
        _cfg.LLM["llama_runner"], _cfg.LLM["model_pte"], _cfg.LLM["tokenizer_path"],
        _cfg.LLM["cpu_cores"], _cfg.LLM["max_new_tokens"], _cfg.LLM["test_mode"],
    )
    advisor.bind_cores()
    advisor.load()
    logger = _log.InspectionLogger()

    events_path = Path(GATING["events_file"])
    decisions_path = Path(GATING["decisions_file"])
    print(f"[Gateway] Watching {events_path} -> logging to CSV & SQLite (Ctrl+C to stop)")

    offset = 0
    with open(decisions_path, "a") as out:
        try:
            while True:
                if not events_path.exists():
                    time.sleep(0.2)
                    continue
                with open(events_path, "r") as f:
                    f.seek(offset)
                    while (line := f.readline()):
                        offset = f.tell()
                        try:
                            ev = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not ev.get("has_defect"):
                            continue
                        hw = {
                            "ram_mb": _bench.read_ram_used_mb(),
                            "temp_c": _bench.read_temp_c(),
                            "watts": _bench.read_power_watts(),
                        }
                        for d in ev.get("detections", []):
                            dec = tiered_response(d["class"], d["confidence"], d["bbox"], advisor)
                            dec["frame"] = ev["frame"]
                            dec["detector_latency_ms"] = ev["latency_ms"]
                            dec["ts"] = time.time()
                            out.write(json.dumps(dec) + "\n")
                            out.flush()
                            logger.log(dec, hw)
                            print(
                                f"[Gateway] {dec['source']:8s} | {dec['defect_type']} | {dec['action']} | "
                                f"det={dec['detector_latency_ms']}ms llm={dec['llm_latency_ms']}ms "
                                f"temp={hw['temp_c']}C"
                            )
                time.sleep(0.2)
        except KeyboardInterrupt:
            print(f"\n[Gateway] Stopped. Total logged inspections: {logger.count()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PS5 Tiered QC Gateway")
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    if args.run:
        run_gateway()
    else:
        p.print_help()
