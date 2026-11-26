#!/usr/bin/env python3
"""Pipeline 2: On-device Llama 3.2 1B-Instruct (4-bit 8da4w) via ExecuTorch + Arm KleidiAI."""
import argparse
import importlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

_cfg = importlib.import_module("01_config")
LLM = _cfg.LLM


class LlamaAdvisor:
    def __init__(self, runner: str, model_pte: str, tokenizer_path: str,
                 cores: list[int], max_new_tokens: int = 48, test_mode: bool = False):
        self.runner = runner
        self.model_pte = model_pte
        self.tokenizer_path = tokenizer_path
        self.cores = cores
        self.max_new_tokens = max_new_tokens
        self.test_mode = test_mode
        self.wake_count = 0
        self.latencies = []

    def bind_cores(self):
        try:
            os.sched_setaffinity(0, set(self.cores))
            print(f"[Llama] Pinned PID {os.getpid()} to cores {self.cores}")
        except Exception as exc:
            print(f"[WARN] Core pinning skipped: {exc}")

    def load(self):
        ready = (
            not self.test_mode
            and Path(self.runner).exists()
            and Path(self.model_pte).exists()
            and Path(self.tokenizer_path).exists()
        )
        if ready:
            self.test_mode = False
            print(f"[Llama] ExecuTorch runner ready: {self.runner} + {self.model_pte}")
        else:
            self.test_mode = True
            print("[Llama] Running in TEST_MODE (llama_main/.pte/tokenizer not found).")

    def advise(self, defect_type: str, confidence: float, bbox: list) -> dict:
        t0 = time.perf_counter()
        self.wake_count += 1

        prompt = (
            "<|start_header_id|>user<|end_header_id|>\n"
            f"Steel QC defect: {defect_type} (conf {confidence:.2f}). "
            "Reply with one compact JSON object: "
            '{"severity":"low|medium|high","recommendation":"short factory fix","action":"stop_line|inspect_manually|alert_supervisor"}'
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{"
        )

        if self.test_mode:
            raw = json.dumps({
                "severity": "high" if confidence >= 0.7 else "medium",
                "recommendation": f"Inspect {defect_type} at {bbox} and verify roller alignment.",
                "action": "inspect_manually",
            })
        else:
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = str(Path(self.runner).parent / "runner")
            cmd = [
                "taskset", "-c", ",".join(map(str, self.cores)),
                self.runner,
                f"--model_path={self.model_pte}",
                f"--tokenizer_path={self.tokenizer_path}",
                f"--cpu_threads={len(self.cores)}",
                f"--max_new_tokens={self.max_new_tokens}",
                f"--prompt={prompt}",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
            raw = res.stdout

        match = re.search(r"\{.*?\}", raw, re.S)
        try:
            advice = json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            advice = {}

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        self.latencies.append(latency_ms)

        return {
            "defect_type": advice.get("defect_type", defect_type),
            "severity": advice.get("severity", "medium"),
            "recommendation": advice.get("recommendation", "Inspect steel strip manually and log batch."),
            "confidence": round(confidence, 2),
            "action": advice.get("action", "inspect_manually"),
            "llm_latency_ms": latency_ms,
        }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PS5 ExecuTorch Llama 3.2 1B Advisor")
    p.add_argument("--test-advice", action="store_true")
    args = p.parse_args()
    adv = LlamaAdvisor(
        LLM["llama_runner"], LLM["model_pte"], LLM["tokenizer_path"],
        LLM["cpu_cores"], LLM["max_new_tokens"], LLM["test_mode"],
    )
    adv.load()
    if args.test_advice:
        print(json.dumps(adv.advise("pitted_surface", 0.68, [40, 50, 120, 130]), indent=2))
    else:
        p.print_help()
