#!/usr/bin/env python3
"""Central configuration for PS5 On-Device Industrial Defect Detection & QC Assistant."""
from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parent
MODELS_DIR    = PROJECT_ROOT / "models"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
RESULTS_DIR   = PROJECT_ROOT / "results"
LOGS_DIR      = PROJECT_ROOT / "logs"

for _d in (MODELS_DIR, TEMPLATES_DIR, RESULTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

HARDWARE = {
    "board": "raspberry_pi_5",
    "camera_index": 0,
    "power_meter_port": "",  # Optional USB power meter (e.g. "/dev/ttyUSB0")
}

# Pipeline 1: Vision Detector (Pure NCNN + Armv8.2 FP16 NEON on Pi 5 Cortex-A76)
DETECTOR = {
    "model_path": str(MODELS_DIR / "yolov8n_ncnn_model"),
    "model_weights": str(MODELS_DIR / "yolov8n.pt"),
    "input_size": 320,
    "confidence_threshold": 0.45,
    "cpu_cores": [0, 1, 2],  # Pinned cores for Vision process
}

# Pipeline 2: Advisory Reasoning (Llama 3.2 1B-Instruct 4-bit via ExecuTorch + KleidiAI)
LLM = {
    "llama_runner": str(Path.home() / "et_build/executorch/cmake-out/examples/models/llama/llama_main"),
    "model_pte": str(MODELS_DIR / "llama3_2_1b_spinquant.pte"),
    "tokenizer_path": str(MODELS_DIR / "tokenizer.model"),
    "cpu_cores": [2, 3],     # Pinned cores for ExecuTorch Llama process
    "max_new_tokens": 48,
    "temperature": 0.1,
    "test_mode": False,      # Automatically falls back to test_mode if runner/.pte missing
}

# Tiered Gating: Fast templates for known high-confidence defects; Llama for novel/uncertain defects
GATING = {
    "event_triggered": True,
    "tiered_response": True,
    "confidence_threshold_llm": 0.75,
    "known_defects": ["crazing", "inclusion", "patches", "scratches", "crack", "scratch", "contamination"],
    "events_file": str(RESULTS_DIR / "events.jsonl"),
    "decisions_file": str(RESULTS_DIR / "decisions.jsonl"),
}

BENCHMARK = {
    "output_csv": str(RESULTS_DIR / "benchmark.csv"),
    "duration_seconds": 300,
    "sample_interval_s": 1.0,
    "measure_temp": True,
    "measure_ram": True,
    "measure_power": True,
}

LOGGING = {
    "csv_path": str(RESULTS_DIR / "inspection_log.csv"),
    "db_path": str(RESULTS_DIR / "inspections.db"),
}


def validate_config():
    print(f"Detector NCNN   : {DETECTOR['model_path']} (exists: {Path(DETECTOR['model_path']).exists()})")
    print(f"Llama runner    : {LLM['llama_runner']} (exists: {Path(LLM['llama_runner']).exists()})")
    print(f"Llama .pte      : {LLM['model_pte']} (exists: {Path(LLM['model_pte']).exists()})")
    print(f"Tokenizer       : {LLM['tokenizer_path']} (exists: {Path(LLM['tokenizer_path']).exists()})")
    print(f"Llama test_mode : {LLM['test_mode']}")


if __name__ == "__main__":
    validate_config()
