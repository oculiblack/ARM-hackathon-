#!/usr/bin/env python3
"""Zero-dependency live Operator QC Console (MJPEG YOLO stream + ExecuTorch Llama advice + telemetry)."""
import importlib
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_cfg = importlib.import_module("01_config")
_bench = importlib.import_module("05_benchmark")

DECISIONS_FILE = Path(_cfg.GATING["decisions_file"])
EVENTS_FILE = Path(_cfg.GATING["events_file"])
CSV_FILE = Path(_cfg.LOGGING["csv_path"])
LIVE_JPG = Path("/dev/shm/ps5_live.jpg")

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PS5 Live QC Console — Raspberry Pi 5</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0b1120;color:#e2e8f0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:18px 24px}
  header{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #1e293b;padding-bottom:12px;margin-bottom:16px}
  h1{font-size:19px;color:#f8fafc} .sub{font-size:12px;color:#94a3b8;margin-top:2px}
  .pill{background:#064e3b;color:#6ee7b7;padding:5px 12px;border-radius:999px;font-size:12px;font-weight:700}
  .kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:16px}
  .kpi{background:#111827;border:1px solid #1e293b;border-radius:10px;padding:12px 14px}
  .kpi .lbl{font-size:11px;text-transform:uppercase;color:#94a3b8}
  .kpi .val{font-size:22px;font-weight:700;color:#38bdf8;margin-top:3px}
  .kpi .val.g{color:#34d399} .kpi .val.a{color:#fbbf24}
  .main{display:grid;grid-template-columns:1.35fr 1fr;gap:16px;margin-bottom:16px;align-items:start}
  .panel{background:#111827;border:1px solid #1e293b;border-radius:10px;padding:14px}
  .ptitle{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:#94a3b8;margin-bottom:10px;display:flex;justify-content:space-between}
  .video{width:100%;border-radius:8px;background:#000;min-height:280px;display:block;object-fit:contain}
  .alert{border-left:5px solid #38bdf8;padding-left:14px;margin-bottom:14px}
  .alert.high,.alert.critical{border-left-color:#ef4444} .alert.medium{border-left-color:#f59e0b} .alert.low{border-left-color:#10b981}
  .badges{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px}
  .b{padding:3px 8px;border-radius:6px;font-size:11px;font-weight:700;text-transform:uppercase}
  .b-high,.b-critical{background:#7f1d1d;color:#fecaca} .b-medium{background:#78350f;color:#fde68a}
  .b-low{background:#064e3b;color:#a7f3d0} .b-llm{background:#1e3a8a;color:#bfdbfe} .b-template{background:#334155;color:#e2e8f0}
  .rec{font-size:15px;color:#f8fafc;line-height:1.45} .meta{font-size:12px;color:#94a3b8;margin-top:6px}
  .det-list{margin-top:12px;border-top:1px solid #1e293b;padding-top:10px}
  .det-item{display:flex;justify-content:space-between;padding:5px 0;font-size:13px;border-bottom:1px solid #1e293b}
  .bar{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
  .btn{background:#0284c7;color:#fff;text-decoration:none;padding:6px 12px;border-radius:6px;font-size:12px;font-weight:600}
  table{width:100%;border-collapse:collapse;background:#111827;border-radius:10px;overflow:hidden;font-size:12.5px}
  th{text-align:left;padding:9px 10px;background:#1e293b;color:#94a3b8;font-size:11px;text-transform:uppercase}
  td{padding:8px 10px;border-top:1px solid #1e293b;color:#cbd5e1}
  @media(max-width:920px){.main,.kpis{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <div>
    <h1>Real-Time Industrial Defect Detection &amp; Quality-Control Assistant</h1>
    <div class="sub">Raspberry Pi 5 (CPU-Only) · Pure NCNN FP16 YOLOv8n (NEU-DET) + ExecuTorch &amp; Arm KleidiAI Llama 3.2 1B</div>
  </div>
  <div class="pill">LIVE · 100% ON-DEVICE</div>
</header>

<div class="kpis">
  <div class="kpi"><div class="lbl">Vision Speed</div><div class="val g" id="fps">-- FPS</div></div>
  <div class="kpi"><div class="lbl">Vision Latency</div><div class="val" id="detms">-- ms</div></div>
  <div class="kpi"><div class="lbl">Template / Llama Alerts</div><div class="val" id="counts">0 / 0</div></div>
  <div class="kpi"><div class="lbl">Pi 5 SoC Temp</div><div class="val a" id="temp">-- °C</div></div>
  <div class="kpi"><div class="lbl">RAM · Throttle</div><div class="val" id="ram">-- MB</div></div>
</div>

<div class="main">
  <div class="panel">
    <div class="ptitle"><span>Live Webcam Stream — YOLO NEU-DET Classification</span><span id="frameNo">Frame #0</span></div>
    <img class="video" src="/video_feed" alt="Live YOLO Webcam Feed">
  </div>
  <div class="panel">
    <div class="ptitle"><span>Operator Advisory (Tiered Template + ExecuTorch Llama 3.2 1B)</span></div>
    <div class="alert" id="alertBox">
      <div class="badges" id="alertBadges"><span class="b b-template">WAITING FOR DEFECTS</span></div>
      <div class="rec" id="alertRec">Point the webcam at a sample to stream live detections...</div>
      <div class="meta" id="alertMeta"></div>
    </div>
    <div class="det-list">
      <div class="ptitle" style="margin-bottom:6px"><span>Latest Frame YOLO Detections</span></div>
      <div id="liveDets" style="font-size:13px;color:#94a3b8">No active detections</div>
    </div>
  </div>
</div>

<div class="bar">
  <h2 style="font-size:14px;color:#f8fafc">Live Inspection Log (On-Device SQLite / CSV)</h2>
  <a class="btn" href="/download-csv">Download inspection_log.csv</a>
</div>
<table>
  <thead><tr><th>Time</th><th>Frame</th><th>Defect</th><th>Severity</th><th>Action</th><th>Source</th><th>Det ms</th><th>LLM ms</th><th>Recommendation</th></tr></thead>
  <tbody id="rows"></tbody>
</table>

<script>
async function refresh(){
  try{
    const d = await (await fetch('/api/state')).json();
    document.getElementById('fps').textContent = d.fps ? d.fps.toFixed(1)+' FPS' : '-- FPS';
    document.getElementById('detms').textContent = d.last_det_ms ? d.last_det_ms.toFixed(1)+' ms' : '-- ms';
    document.getElementById('counts').textContent = `${d.template_count} / ${d.llm_count}`;
    document.getElementById('temp').textContent = d.temp_c > 0 ? d.temp_c.toFixed(1)+' °C' : '--';
    document.getElementById('ram').textContent = `${Math.round(d.ram_mb)} MB · ${d.throttled.replace('throttled=','')}`;
    document.getElementById('frameNo').textContent = 'Frame #' + (d.last_frame || 0);

    const ld = document.getElementById('liveDets');
    if(d.live_detections && d.live_detections.length){
      ld.innerHTML = d.live_detections.map(x =>
        `<div class="det-item"><b>${x.class}</b><span>${(x.confidence*100).toFixed(1)}% · [${x.bbox.join(', ')}]</span></div>`
      ).join('');
    } else { ld.textContent = 'No defects detected in current frame'; }

    if(d.latest){
      const L = d.latest, sev = (L.severity||'medium').toLowerCase();
      document.getElementById('alertBox').className = 'alert ' + sev;
      const sc = L.source === 'llm' ? 'b-llm' : 'b-template';
      const st = L.source === 'llm' ? 'ExecuTorch Llama 3.2 1B' : 'Fast Template';
      document.getElementById('alertBadges').innerHTML =
        `<span class="b b-${sev}">${sev}</span><span class="b ${sc}">${st}</span>` +
        `<span class="b b-template">${L.defect_type} (${Math.round((L.confidence||0)*100)}%)</span>` +
        `<span class="b b-${sev}">${L.action}</span>`;
      document.getElementById('alertRec').textContent = L.recommendation || 'Inspect manually.';
      document.getElementById('alertMeta').textContent =
        `Frame #${L.frame} · Vision: ${L.detector_latency_ms} ms · Advisory: ${L.llm_latency_ms} ms`;
    }

    document.getElementById('rows').innerHTML = d.recent.map(x => {
      const t = new Date((x.ts||0)*1000).toLocaleTimeString(), sev=(x.severity||'medium').toLowerCase();
      return `<tr><td>${t}</td><td>${x.frame??'-'}</td><td><b>${x.defect_type}</b></td>` +
        `<td><span class="b b-${sev}">${sev}</span></td><td>${x.action}</td><td>${x.source}</td>` +
        `<td>${x.detector_latency_ms??'-'}</td><td>${x.llm_latency_ms??'-'}</td>` +
        `<td>${(x.recommendation||'').slice(0,70)}</td></tr>`;
    }).join('');
  }catch(e){}
}
refresh(); setInterval(refresh, 750);
</script>
</body>
</html>
"""


def tail_jsonl(path: Path, n: int = 200):
    if not path.exists():
        return []
    out = []
    for ln in path.read_text(errors="ignore").splitlines()[-n:]:
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            b = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif self.path == "/video_feed":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    if LIVE_JPG.exists():
                        jpg = LIVE_JPG.read_bytes()
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
                        )
                    time.sleep(0.05)
            except Exception:
                pass
        elif self.path == "/api/state":
            decs = tail_jsonl(DECISIONS_FILE, 200)
            evs = tail_jsonl(EVENTS_FILE, 30)
            lats = [e["latency_ms"] for e in evs if e.get("latency_ms", 0) > 0]
            avg = sum(lats) / len(lats) if lats else 0.0
            last_ev = evs[-1] if evs else {}
            b = json.dumps({
                "fps": (1000.0 / avg) if avg > 0 else 0.0,
                "last_det_ms": lats[-1] if lats else 0.0,
                "last_frame": last_ev.get("frame", 0),
                "live_detections": last_ev.get("detections", []),
                "template_count": sum(1 for d in decs if d.get("source") == "template"),
                "llm_count": sum(1 for d in decs if d.get("source") == "llm"),
                "temp_c": _bench.read_temp_c(),
                "ram_mb": _bench.read_ram_used_mb(),
                "throttled": _bench.read_throttled(),
                "latest": decs[-1] if decs else None,
                "recent": list(reversed(decs[-15:])),
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif self.path == "/download-csv":
            b = CSV_FILE.read_bytes() if CSV_FILE.exists() else b"No logs yet\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Disposition", "attachment; filename=inspection_log.csv")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    print("[Dashboard] Live Operator Console running at http://0.0.0.0:8080")
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
