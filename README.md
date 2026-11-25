# PS5: Winning Design — Bharat AI-SoC Challenge 2026–27

**Track:** PS5 — Real-Time Industrial Defect Detection & Quality-Control Assistant (Manufacturing & Industry 4.0)  
**Target Hardware:** Raspberry Pi 5 (CPU-only, or Pi 5 + AI HAT+ / AI HAT+ 2)  
**Compulsory Stack:** Llama family model (Llama 3.2 1B/3B-Instruct) + ExecuTorch runtime + Arm-based Linux device + on-device, no cloud  
**Goal:** Highest probability of winning by closing 7 research gaps identified from a survey of 277 papers.

---

## 🎯 Why This Design Wins

The competition theme is **"engineering AI-enabled products on SoCs"** — optimisation, system integration, edge deployment, benchmarking, validation. Most academic papers report accuracy only; **none** report RAM + ms + Watts together on the same Pi 5 board (Gap #6, High confidence). No paper combines a quantized small LLM operator assistant with an edge defect detector end-to-end on Pi 5 (Gap #7).

| Gap | What We Close | Competition Impact |
|-----|---------------|-------------------|
| #1 | Event-triggered LLM wake-on-defect with measured mJ/inspection savings | Shows systems thinking, not just model accuracy |
| #2 | Tiered response: templates for known defects, Llama only for novel/ambiguous | Demonstrates practical deployment awareness |
| #3 | Constrained/forced JSON decoding for Llama on Arm | Proves LLM output reliability for operator alerts |
| #4 | CPU core partitioning: vision on big cores, LLM on little cores | Shows SoC optimisation (the competition theme) |
| #5 | INT8 + coreset sweep vs accuracy on Pi 5 | Quantifiable efficiency trade-offs |
| #6 | RAM + ms + Watts measured together on same Pi 5 | **Highest differentiator** — no competitor will have this |
| #7 | End-to-end Llama + detector on Pi 5, full benchmark | Directly addresses the compulsory requirements with numbers |

---

## 🏗️ System Architecture

```
                    Raspberry Pi 5 (Arm Cortex-A76)
  ┌─────────────────────────────────────────────────────────────────┐
  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
  │  │  CPU Core 0-3│  │  CPU Core 4-5│  │   AI HAT+ (Hailo)  │   │
  │  │  (Big cores) │  │  (Little core│  │   (optional NPU)   │   │
  │  │              │  │   LLM only   │  │                    │   │
  │  │   DETECTOR   │◄─┤   LLAMA      │  │                    │   │
  │  │   YOLOv8n    │  │   ExecuTorch │  │                    │   │
  │  │   (INT8)     │  │   4-bit      │  │                    │   │
  │  └──────────────┘  └──────────────┘  └────────────────────┘   │
  │         │                   │                   │              │
  │         ▼                   ▼                   ▼              │
  │  Continuous scan        On defect → LLM     HailoRT (if HAT+) │
  │  ~10-50 FPS             generates JSON      accelerates YOLO   │
  │         │                   │                   │              │
  │         └───────────────────┴───────────────────┘              │
  │                         │                                       │
  │              ┌──────────▼────────────┐                          │
  │              │   Operator Alert +    │                          │
  │              │   Log (JSON, CSV)     │                          │
  │              │   (No cloud, on-device)│                         │
  │              └───────────────────────┘                          │
  └─────────────────────────────────────────────────────────────────┘
```

**Data flow:**
1. Camera captures frame → Detector (YOLOv8n INT8) runs continuously on big cores
2. If no defect → log frame, skip, continue (Llama stays asleep)
3. If defect → wake Llama on little core(s) → Llama classifies type/severity + recommendation → output forced JSON
4. Display alert to operator → append to inspection log (CSV/SQLite on SD card)

---

## 📦 Hardware Bill of Materials

| Item | Specification | Qty | Purpose |
|------|--------------|-----|---------|
| Raspberry Pi 5 | 4 GB or 8 GB, Raspberry Pi OS 64-bit | 1 | Main compute |
| Camera Module 3 | 12 MP, CSI/USB | 1 | Defect image input |
| AI HAT+ / AI HAT+ 2 (optional) | Hailo-8L NPU | 1 | NPU-accelerated detection (optional) |
| USB Power Meter | 0–60 V, 0–10 A, Wh logging | 1 | Measure Watts/Joules (Gap #6) |
| microSD Card | 64 GB+, A2 class | 1 | OS, models, logs |
| Heatsink + Fan | Pi 5 active cooling | 1 | Prevent thermal throttling |
| Industrial Test Samples | PCB, steel, fabric defects | Set | Demo dataset |

---

## 💻 Software Stack

| Layer | Component | Version | Role |
|-------|-----------|---------|------|
| OS | Raspberry Pi OS (64-bit) | Latest | Host OS |
| Runtime | ExecuTorch | Latest | Llama inference on Arm CPU |
| ML Framework | PyTorch | 2.x | Model export/quantization |
| Detector | YOLOv8n / YOLO11n | Ultralytics | Surface defect detection |
| Quantization | INT8 PTQ / AWQ | — | Compress detector + Llama |
| LLM | Llama 3.2 1B-Instruct | — | Advisory/reasoning (compulsory) |
| Decoding | Constrained JSON | — | Forced structured output |
| Camera | Picamera2 / OpenCV | — | Frame capture |
| Logging | SQLite / CSV | — | Inspection results log |
| Benchmarking | vcgencmd, free, custom | — | Latency/RAM/Temp/Power |

---

## 📁 File Layout

```
ps5_winning_design/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── 00_setup_pi5.sh                # One-time system setup (run once, reboot)
├── 01_config.py                   # Central config (paths, thresholds, cores)
├── 02_detector.py                 # YOLOv8n INT8 defect detector
├── 03_llama_advisor.py            # Llama 3.2 1B ExecuTorch advisor
├── 04_gateway.py                  # Event-triggered gating + tiered response
├── 05_benchmark.py                # Latency/RAM/Temp/Power sampler
├── 06_logger.py                   # CSV + SQLite inspection logger
├── 07_run_pipeline.py             # Main orchestrator
├── models/                        # yolov8n.pt, yolov8n_openvino_model/, llama .ptl
├── templates/                     # crack.json, scratch.json, contamination.json
├── results/                       # events.jsonl, decisions.jsonl, benchmark.csv, logs
└── logs/                          # additional runtime logs
```

---

## 🚀 Step-by-Step Setup

### Step 0 — One-time Pi 5 Setup

```bash
# On the Raspberry Pi 5 (Raspberry Pi OS 64-bit)
cd ~
git clone <your-repo> ps5_winning_design   # or copy files manually
cd ps5_winning_design
chmod +x 00_setup_pi5.sh
bash 00_setup_pi5.sh
sudo reboot                                 # REQUIRED for isolcpus=4,5
```

After reboot:
```bash
source ~/ps5venv/bin/activate
cd ~/ps5_winning_design
```

### Step 1 — Validate Configuration

```bash
python3 01_config.py
```

Expected output: prints all paths and whether model files exist.

### Step 2 — Export Models

**Detector (YOLOv8n → OpenVINO INT8):**
```bash
python3 -c "
from ultralytics import YOLO
m = YOLO('yolov8n.pt')
m.export(format='openvino', int8=True, imgsz=640)
"
# Output goes to ./yolov8n_openvino_model/ — copy to models/
mv yolov8n_openvino_model models/
```

**Llama (Llama 3.2 1B → ExecuTorch 4-bit .ptl):**
Follow ExecuTorch docs to export + quantize. Place at:
```
models/llama-3.2-1b-instruct-q4_k_m.ptl
```
If you don't have the `.ptl` yet, set `LLM.test_mode = True` in `01_config.py` — the advisor will run in fake-inference mode so you can still debug the full pipeline.

### Step 3 — Test Each Module Individually

Run these **in order**. Each prints `[TEST] ... PASSED` on success.

| Command | What it tests |
|---------|---------------|
| `python3 02_detector.py --test-camera` | Camera opens and grabs frames |
| `python3 02_detector.py --test-model` | Model loads + dummy inference |
| `python3 03_llama_advisor.py --test-advice` | Advice JSON generation (no model needed) |
| `python3 03_llama_advisor.py --test-load` | Tries loading real `.ptl` model |
| `python3 04_gateway.py --test-tiered` | Template vs LLM routing logic |
| `python3 05_benchmark.py --test-sensors` | Reads temp/RAM/power/throttle |
| `python3 06_logger.py --test-log` | Writes + reads CSV/SQLite |
| `python3 07_run_pipeline.py --test-all` | Runs all of the above |

### Step 4 — Run the Full Pipeline

```bash
python3 07_run_pipeline.py --run
```

This starts 3 threads:
1. **Detector** — continuous capture on camera, writes `events.jsonl`
2. **Gateway** — reads events, applies tiered response, writes `decisions.jsonl`, logs to CSV/SQLite
3. **Benchmark** — samples temp/RAM/power every second, writes `benchmark.csv`

Press **Ctrl+C** to stop all threads gracefully.

### Step 5 — Collect Results for Competition

After running, check:
```bash
ls -la results/
cat results/benchmark.csv
cat results/decisions.jsonl
cat results/inspection_log.csv
```

Fill these numbers into the technical report template (`PS5_Final_Project_Report.docx`).

---

## ⚡ Optimization Techniques

| Technique | What It Does | Gap Closed |
|-----------|-------------|------------|
| INT8 Quantization (Detector) | YOLOv8n exported to OpenVINO INT8; reduces model size and inference latency on Arm CPU | #5 |
| 4-bit LLM Quantization | Llama 3.2 1B-Instruct quantized to 4-bit (AWQ/INT4) for ExecuTorch; fits Pi 5 RAM | #5 |
| Event-Triggered Gating | Llama remains asleep until the detector signals a defect; eliminates continuous LLM power draw | #1 |
| Tiered Response | Known defect classes use pre-written JSON templates (zero LLM latency); novel/ambiguous cases wake Llama | #2 |
| Constrained Decoding | Forced-JSON output schema guarantees parseable operator recommendations | #3 |
| CPU Core Partitioning | isolcpus + taskset isolates vision and LLM threads on separate core clusters | #4 |
| Performance Governor | CPU governor locked to 'performance' to eliminate frequency-scaling variance | — |
| Thermal Throttle Monitoring | vcgencmd get_throttled and measure_temp sampled each second; auto-pauses above 80°C | — |

---

## 📊 Benchmark & Results

The following table must be populated with measured values from your Pi 5 benchmark run (`results/benchmark.csv`). Bracketed values are placeholders — replace with actual numbers before submission.

| Metric | Measured Value | Hardware Context | Target |
|--------|---------------|-----------------|--------|
| Detector FPS (INT8) | [INSERT FPS] | Pi 5 CPU-only | Real-time (≥ target FPS) |
| Detector p95 Latency | [INSERT ms] | Pi 5 big cores (0–3) | < 100 ms |
| Llama Time to First Token | [INSERT ms] | Pi 5 little cores (4–5) | < 500 ms |
| Llama Tokens/s | [INSERT tok/s] | 4-bit ExecuTorch | Usable for operator advice |
| Peak RAM (Detector) | [INSERT MB] | Pi 5 | < 500 MB |
| Peak RAM (Llama) | [INSERT MB] | Pi 5 | < 1 GB |
| Max Temperature | [INSERT °C] | Pi 5 SoC | < 80 °C |
| Throttle Events | [INSERT count] | Pi 5 | 0 during benchmark |
| Idle Power | [INSERT W] | Pi 5 + USB meter | Baseline |
| Defect + Llama Power | [INSERT W] | Pi 5 + USB meter | Peak load |
| Energy per Inspection | [INSERT mJ] | Pi 5 | Lower than continuous Llama |

**Evidence files to submit:**
- `results/benchmark.csv` — time-series temperature, RAM, power, latency
- `results/decisions.jsonl` — every operator decision (template vs LLM source, severity, action)
- `results/inspection_log.csv` / `inspections.db` — full inspection trace
- Demo video: defect detected → Llama wakes → operator recommendation + local log, all on-device

---

## ⚠️ Risk Assessment & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Thermal overheating / emergency shutdown | Medium | High | Active heatsink + fan; auto-pause at 80°C; resume at 70°C |
| 4 GB RAM OOM with Llama + detector | Medium | High | 4-bit quantization; run detector-only if needed; close background services |
| Camera device busy on restart | Low | Medium | try/finally around VideoCapture; release on SIGTERM |
| SQLite database locked under concurrency | Low | Medium | WAL mode + threading lock for DB writes |
| SD card wear from frequent JSONL/CSV writes | Medium | Low | Rotate logs at 50 MB; batch writes; sample benchmark at 1 Hz |
| Llama inference hang stalls pipeline | Low | Medium | Watchdog thread; timeout; degrade to detector-only |

---

## 📤 GitHub & Open-Source Strategy

**Should you put everything on GitHub? Yes — this significantly increases win probability.**

### Why GitHub helps:
Judges can verify the code compiles and runs on Pi 5. A public repository demonstrates engineering rigor, transparency, and reproducibility — key criteria for an SoC challenge.

### What to publish:
- All Python source, `config.yaml`, model export scripts, `requirements.txt`
- `results/` folder (benchmark CSV, decisions JSONL), demo video link, technical report

### What NOT to publish:
- Do NOT publish proprietary industrial test images if they are confidential; use synthetic or publicly available defect datasets (MVTec AD, NEU-DET, KolektorSDD2) for the public repo.

---

## 🐞 Common Errors & Fixes

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `No module named 'ultralytics'` | pip package not installed | `pip install ultralytics` (in venv) |
| `Cannot open camera index 0` | Camera not enabled or in use | `sudo raspi-config` → enable camera; close other apps |
| `model not found` | `.pt` or `.ptl` missing | Export YOLO/OpenVINO; download Llama `.ptl`; or set `test_mode=True` |
| `vcgencmd: command not found` | Not on Pi OS or VideoCore missing | Temp reading returns -1; OK for testing, add vcgencmd for real benchmarks |
| `Permission denied` on `/dev/ttyUSB0` | USB power meter permissions | `sudo chmod 666 /dev/ttyUSB0` or add user to dialout group |
| Low FPS / thermal throttling | CPU governor not performance | `echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` |
| Import error on `01_config.py` | Filename starts with digit | All files use `importlib.import_module("01_config")` — do not change to normal `import` |

---

## ✅ Winning Checklist (Competition Submission)

- [ ] `00_setup_pi5.sh` ran, Pi 5 rebooted
- [ ] All `--test-*` commands print PASSED
- [ ] Full pipeline runs 5+ minutes without crashing
- [ ] `results/benchmark.csv` has temp + RAM + power columns
- [ ] `results/decisions.jsonl` shows both `template` and `llm` sources
- [ ] Demo video recorded: defect → Llama wake → operator alert + log
- [ ] Source code on GitHub (public)
- [ ] Technical report filled with your measured numbers
- [ ] Repo URL in your submission form

---

## 📄 Technical Report

See `PS5_Final_Project_Report.docx` — an editable Word document containing the full technical report. Fill in all `[INSERT ...]` placeholders with your measured benchmark numbers before submission.

---

*Built for the Bharat AI-SoC Challenge 2026–27, PS5 Track — Manufacturing & Industry 4.0*
