# Real-Time Industrial Defect Detection & Quality-Control Assistant (PS5)

**Bharat AI-SoC Challenge 2026–27 · Problem Statement 5 (Manufacturing & Industry 4.0)**  
100% on-device, zero-cloud industrial visual inspection and quality-control advisory system running on a **CPU-only Raspberry Pi 5 (8 GB, Arm Cortex-A76)**.

## Architecture (Two Strictly Separated Runtimes)

1. **Vision Pipeline (`02_detector.py` — Pure NCNN + Armv8.2 FP16 NEON):**  
   Runs a custom **NEU-DET** fine-tuned YOLOv8n (`320×320`) detector in pure C++/Python NCNN (zero PyTorch runtime overhead), pinned to dedicated CPU cores via `os.sched_setaffinity`, streaming annotated MJPEG frames via Linux RAM disk (`/dev/shm`) and emitting structured JSON defect records.
2. **Advisory Pipeline (`03_llama_advisor.py` + `04_gateway.py` — ExecuTorch + Arm KleidiAI):**  
   Event-triggered, tiered response gateway pinned to Cores `2–3`. High-confidence known defects (`crazing`, `inclusion`, `patches`, `scratches`) trigger instant deterministic templates (`0 ms` LLM wake), while novel or uncertain defects (`pitted_surface`, `rolled-in_scale`, or `< 0.75` confidence) wake **4-bit `8da4w` Llama 3.2 1B-Instruct** running natively via ExecuTorch's C++ `llama_main` runner with **Arm KleidiAI** NEON kernels.
3. **Operator UI & Local Logging (`07_dashboard.py`, `05_benchmark.py`, `06_logger.py`):**  
   Zero-dependency live web console (`http://<pi-ip>:8080`) displaying the real-time YOLO webcam stream, severity badges, actionable factory advice, Pi 5 SoC temperature/RAM/throttling telemetry, and downloadable local SQLite/CSV logs.

## Measured On-Device Benchmarks (Raspberry Pi 5, CPU-Only)

| Pipeline | Runtime & Precision | Pinned Cores | Measured Performance |
| :--- | :--- | :--- | :--- |
| **Vision (NEU-DET)** | Pure NCNN (`320×320`, Armv8.2 FP16 NEON) | Cores `0–1` (baseline) / `0–2` | **53.52 FPS**, **18.61 ms P50**, **19.08 ms P95** |
| **Advisory (Llama 3.2 1B)** | ExecuTorch `1.6.0` + XNNPACK + Arm KleidiAI (`8da4w`, 765 MB `.pte`) | Cores `2–3` (`--cpu_threads=2`) | **82.07 tok/s prefill**, **18.15 tok/s decode**, **329 ms TTFT**, **774 MiB RSS** |
| **Thermals & Stability** | `vcgencmd measure_temp` / `get_throttled` | All 4 Cortex-A76 cores | **45.5–47.7 °C**, `throttled=0x0` |

## Repository Structure

```text
.
├── 00_setup_pi5.sh                  # One-time Raspberry Pi 5 setup
├── 01_config.py                     # Central paths, core affinity, gating thresholds
├── 02_detector.py                   # Pure NCNN FP16 NEU-DET detector + /dev/shm MJPEG writer
├── 03_llama_advisor.py              # ExecuTorch C++ llama_main wrapper + JSON parser
├── 04_gateway.py                    # Event-triggered tiered template/LLM router + logger
├── 05_benchmark.py                  # Pi 5 temp, RAM, throttling, and USB power meter harness
├── 06_logger.py                     # Append-only local SQLite + CSV inspection logger
├── 07_dashboard.py                  # Live Operator Web Console (port 8080)
├── start_ps5.sh                     # One-command launcher for all 3 live services
├── requirements.txt                 # Minimal runtime Python dependencies
├── templates/                       # Fast-path JSON templates for known steel defects
└── colab/
    ├── train_neudet_yolov8n.py      # Reproducible NEU-DET YOLOv8n training & NCNN export
    └── export_llama32_1b_executorch.py  # ExecuTorch 4-bit (8da4w) Llama 3.2 1B .pte export
```

## Quick Start on Raspberry Pi 5

```bash
chmod +x 00_setup_pi5.sh start_ps5.sh
./00_setup_pi5.sh
# Place your exported models in models/:
#   models/yolov8n_ncnn_model/
#   models/llama3_2_1b_spinquant.pte
#   models/tokenizer.model
./start_ps5.sh
# Open http://<pi-ip>:8080 in your browser
```

## References
- [Bharat AI-SoC Challenge 2026–27](https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html)
- [NEU Surface Defect Database (Kaggle)](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)
- [ExecuTorch + Arm KleidiAI](https://pytorch.org/blog/bringing-generative-ai-to-the-masses-with-executorch-and-kleidiai/)
- [Run Llama 3 on Raspberry Pi 5 using ExecuTorch (Arm Learning Paths)](https://learn.arm.com/learning-paths/embedded-and-microcontrollers/rpi-llama3/llama3/)
