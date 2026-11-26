#!/usr/bin/env python3
"""Pipeline 1: Real-time NEU-DET steel defect detector (Pure NCNN + Armv8.2 FP16 NEON)."""
import argparse
import importlib
import json
import os
import signal
import time
from pathlib import Path

import cv2
import ncnn
import numpy as np
import yaml

_cfg = importlib.import_module("01_config")
DETECTOR = _cfg.DETECTOR
RESULTS_DIR = _cfg.RESULTS_DIR

cv2.setNumThreads(1)
DEFAULT_NAMES = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches",
}


class DefectDetector:
    def __init__(self, model_dir: str, conf: float, input_size: int, cores: list[int], output_file: str):
        self.model_dir = Path(model_dir)
        self.conf = conf
        self.input_size = input_size
        self.cores = cores
        self.output_file = output_file
        self.net = None
        self.in_name = "in0"
        self.out_name = "out0"
        self.names = DEFAULT_NAMES
        self.frame_count = 0
        self.defect_count = 0
        self.latencies = []
        self._stop = False

    def bind_cores(self):
        try:
            os.sched_setaffinity(0, set(self.cores))
            print(f"[Detector] Pinned PID {os.getpid()} to cores {self.cores}")
        except Exception as exc:
            print(f"[WARN] Core pinning skipped: {exc}")

    def load(self):
        param = self.model_dir / "model.ncnn.param"
        binf = self.model_dir / "model.ncnn.bin"
        meta = self.model_dir / "metadata.yaml"
        if not param.exists() or not binf.exists():
            raise FileNotFoundError(
                f"NCNN model files not found in {self.model_dir}. "
                "Export best.pt using colab/train_neudet_yolov8n.py first."
            )
        if meta.exists():
            data = yaml.safe_load(meta.read_text()) or {}
            if isinstance(data.get("names"), dict):
                self.names = {int(k): v for k, v in data["names"].items()}

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = len(self.cores)
        self.net.opt.use_fp16_packed = True
        self.net.opt.use_fp16_storage = True
        self.net.opt.use_fp16_arithmetic = True
        self.net.opt.use_packing_layout = True
        self.net.load_param(str(param))
        self.net.load_model(str(binf))
        self.in_name = self.net.input_names()[0]
        self.out_name = self.net.output_names()[0]
        print(f"[Detector] Pure NCNN FP16 loaded: {self.model_dir} | classes={list(self.names.values())}")

    def infer(self, frame: np.ndarray):
        t0 = time.perf_counter()
        h0, w0 = frame.shape[:2]
        sz = self.input_size
        mat_in = ncnn.Mat.from_pixels_resize(frame, ncnn.Mat.PixelType.PIXEL_BGR2RGB, w0, h0, sz, sz)
        mat_in.substract_mean_normalize([], [1 / 255.0, 1 / 255.0, 1 / 255.0])

        ex = self.net.create_extractor()
        ex.input(self.in_name, mat_in)
        _, mat_out = ex.extract(self.out_name)
        out = np.array(mat_out)
        if out.ndim == 2 and out.shape[0] < out.shape[1]:
            out = out.T

        scores = out[:, 4:]
        cls_ids = np.argmax(scores, axis=1)
        confs = scores[np.arange(len(scores)), cls_ids]
        mask = confs >= self.conf

        detections = []
        if np.any(mask):
            boxes_xywh = out[mask, :4]
            f_confs = confs[mask]
            f_cls = cls_ids[mask]
            sx, sy = w0 / sz, h0 / sz
            boxes_cv = []
            for cx, cy, bw, bh in boxes_xywh:
                x1 = int((cx - bw / 2) * sx)
                y1 = int((cy - bh / 2) * sy)
                boxes_cv.append([x1, y1, int(bw * sx), int(bh * sy)])
            keep = cv2.dnn.NMSBoxes(boxes_cv, f_confs.tolist(), self.conf, 0.45)
            for i in np.array(keep).flatten():
                x, y, w, h = boxes_cv[i]
                detections.append({
                    "class": self.names.get(int(f_cls[i]), str(f_cls[i])),
                    "confidence": round(float(f_confs[i]), 3),
                    "bbox": [x, y, x + w, y + h],
                })
                self.defect_count += 1

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.latencies.append(latency_ms)
        self.frame_count += 1
        return detections, latency_ms

    def get_stats(self):
        if not self.latencies:
            return {"frames": 0, "defects": 0}
        return {
            "frames": self.frame_count,
            "defects": self.defect_count,
            "latency_p50_ms": round(float(np.percentile(self.latencies, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(self.latencies, 95)), 2),
            "avg_fps": round(self.frame_count / max(sum(self.latencies) / 1000.0, 1e-3), 2),
        }

    def run(self, source: str = "0", max_frames: int | None = None):
        self.bind_cores()
        src = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera/video source: {source}")
        signal.signal(signal.SIGINT, lambda s, f: setattr(self, "_stop", True))

        with open(self.output_file, "a") as out:
            while not self._stop:
                ret, frame = cap.read()
                if not ret:
                    break
                detections, latency_ms = self.infer(frame)

                # Update live RAM-disk MJPEG frame every 3rd frame to avoid CPU overhead
                if self.frame_count % 3 == 0:
                    vis = frame.copy()
                    for d in detections:
                        x1, y1, x2, y2 = d["bbox"]
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 60, 255), 2)
                        lbl = f"{d['class']} {d['confidence']*100:.0f}%"
                        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                        cv2.rectangle(vis, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), (0, 60, 255), -1)
                        cv2.putText(vis, lbl, (x1 + 4, max(16, y1 - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    fps_now = 1000.0 / max(latency_ms, 0.1)
                    cv2.putText(
                        vis,
                        f"Pi5 NCNN FP16 | {fps_now:.1f} FPS ({latency_ms:.1f} ms) | Defects: {len(detections)}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2,
                    )
                    ok, jpg = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    if ok:
                        Path("/dev/shm/ps5_live.tmp").write_bytes(jpg.tobytes())
                        os.replace("/dev/shm/ps5_live.tmp", "/dev/shm/ps5_live.jpg")

                has_defect = len(detections) > 0
                event = {
                    "ts": time.time(),
                    "frame": self.frame_count,
                    "latency_ms": round(latency_ms, 2),
                    "has_defect": has_defect,
                    "detections": detections,
                }
                out.write(json.dumps(event) + "\n")
                out.flush()
                if has_defect:
                    print(f"[Detector] DEFECT frame={self.frame_count} lat={latency_ms:.1f}ms "
                          f"n={len(detections)} -> {detections[0]['class']}")
                if max_frames and self.frame_count >= max_frames:
                    break

        cap.release()
        print("[Detector] Stats:", json.dumps(self.get_stats(), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PS5 Pure NCNN NEU-DET Defect Detector")
    p.add_argument("--run", action="store_true")
    p.add_argument("--source", default="0", help="Camera index (0) or video path")
    p.add_argument("--frames", type=int, default=None, help="Optional max frames")
    args = p.parse_args()
    if args.run:
        det = DefectDetector(
            DETECTOR["model_path"], DETECTOR["confidence_threshold"],
            DETECTOR["input_size"], DETECTOR["cpu_cores"], str(RESULTS_DIR / "events.jsonl"),
        )
        det.load()
        det.run(source=args.source, max_frames=args.frames)
    else:
        p.print_help()
