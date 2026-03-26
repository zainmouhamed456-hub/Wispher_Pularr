from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import psutil
except ImportError:
    psutil = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple training progress dashboard for Whisper Pularr runs.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--refresh-seconds", type=int, default=20)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _scaled_metric(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number * 100.0 if number <= 1.0 else number


def _pseudo_report_path(artifacts_dir: Path) -> Path | None:
    candidates = [
        artifacts_dir / "pseudo_labels.report.json",
        artifacts_dir / "pseudo_labels.jsonl.report.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _collect_epoch_metrics(run_dir: Path) -> list[dict[str, Any]]:
    full_validation_dir = run_dir / "full_validation"
    if not full_validation_dir.is_dir():
        return []
    metrics: list[dict[str, Any]] = []
    for path in sorted(full_validation_dir.glob("epoch_*.json")):
        payload = _read_json(path)
        if not payload:
            continue
        raw_metrics = payload.get("metrics") or {}
        epoch_text = path.stem.rsplit("_", 1)[-1]
        epoch = int(epoch_text) if epoch_text.isdigit() else epoch_text
        metrics.append(
            {
                "epoch": epoch,
                "normalized_wer": _scaled_metric(raw_metrics.get("normalized_wer")),
                "normalized_cer": _scaled_metric(raw_metrics.get("normalized_cer")),
                "raw_wer": _scaled_metric(raw_metrics.get("raw_wer")),
                "raw_cer": _scaled_metric(raw_metrics.get("raw_cer")),
                "sample_count": payload.get("sample_count"),
                "path": str(path),
            }
        )
    return metrics


def _is_active_run(run_dir: Path) -> bool:
    if psutil is None:
        return False
    run_dir_str = str(run_dir)
    for process in psutil.process_iter(["cmdline"]):
        try:
            cmdline = process.info.get("cmdline") or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if not cmdline:
            continue
        joined = " ".join(cmdline)
        if "train.py" in joined and run_dir_str in joined:
            return True
    return False


def _summarize_run(run_dir: Path, label: str | None = None) -> dict[str, Any]:
    run_summary = _read_json(run_dir / "run_summary.json") or {}
    best_metrics = run_summary.get("best_metrics") or {}
    epoch_metrics = _collect_epoch_metrics(run_dir)
    latest_epoch = epoch_metrics[-1] if epoch_metrics else None
    is_active = _is_active_run(run_dir)
    return {
        "name": label or run_dir.name,
        "run_dir": str(run_dir),
        "stage": run_summary.get("stage"),
        "preset": run_summary.get("preset"),
        "status": "completed" if run_summary else ("running" if (is_active or epoch_metrics) else "pending"),
        "best_epoch": run_summary.get("best_epoch"),
        "best_model_dir": run_summary.get("best_model_dir") or str(run_dir / "best_full_eval"),
        "normalized_wer": _scaled_metric(best_metrics.get("normalized_wer")),
        "normalized_cer": _scaled_metric(best_metrics.get("normalized_cer")),
        "raw_wer": _scaled_metric(best_metrics.get("raw_wer")),
        "raw_cer": _scaled_metric(best_metrics.get("raw_cer")),
        "latest_normalized_wer": latest_epoch.get("normalized_wer") if latest_epoch else None,
        "latest_normalized_cer": latest_epoch.get("normalized_cer") if latest_epoch else None,
        "latest_raw_wer": latest_epoch.get("raw_wer") if latest_epoch else None,
        "latest_raw_cer": latest_epoch.get("raw_cer") if latest_epoch else None,
        "latest_epoch": latest_epoch,
        "epoch_metrics": epoch_metrics,
    }


def _collect_supervised_runs(root: Path) -> list[dict[str, Any]]:
    runs_root = root / "runs"
    results: list[dict[str, Any]] = []
    for trial_name in ("trial_a", "trial_b", "trial_c"):
        run_dir = runs_root / trial_name
        if run_dir.exists():
            results.append(_summarize_run(run_dir, label=trial_name))
    return results


def _collect_self_train_runs(root: Path) -> dict[str, Any]:
    output_root = root / "runs" / "self_train_snapshots"
    sequence_summary = _read_json(output_root / "sequence_summary.json") or {}
    sequence_runs = sequence_summary.get("runs") or []
    manifests_done = {Path(item.get("run_dir", "")).name for item in sequence_runs}

    runs: list[dict[str, Any]] = []
    if output_root.is_dir():
        for run_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
            run_data = _summarize_run(run_dir)
            run_data["in_sequence_summary"] = run_dir.name in manifests_done
            runs.append(run_data)

    current_run = None
    for run in runs:
        if run["status"] == "running":
            current_run = run
            break
    if current_run is None and runs:
        current_run = runs[-1]

    return {
        "output_root": str(output_root),
        "sequence_summary_path": str(output_root / "sequence_summary.json"),
        "last_best_full_eval_dir": sequence_summary.get("last_best_full_eval_dir"),
        "current_run": current_run,
        "runs": runs,
    }


def collect_dashboard_data(root: Path) -> dict[str, Any]:
    artifacts_dir = root / "artifacts"
    pseudo_report_path = _pseudo_report_path(artifacts_dir)
    pseudo_report = _read_json(pseudo_report_path) if pseudo_report_path else None
    manifest_dir = artifacts_dir / "pseudo_labels_manifests"
    manifest_paths = sorted(manifest_dir.glob("*.jsonl")) if manifest_dir.is_dir() else []
    supervised_runs = _collect_supervised_runs(root)
    self_train = _collect_self_train_runs(root)

    return {
        "root": str(root),
        "pseudo_labeling": {
            "report_path": str(pseudo_report_path) if pseudo_report_path else None,
            "report": pseudo_report,
            "manifest_dir": str(manifest_dir),
            "manifest_count": len(manifest_paths),
            "manifest_names": [path.name for path in manifest_paths],
        },
        "supervised_runs": supervised_runs,
        "self_train": self_train,
    }


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Whisper Pularr Dashboard</title>
  <style>
    :root {
      --bg: #f3efe6;
      --paper: #fffaf2;
      --ink: #1f1b16;
      --muted: #74695d;
      --line: #d6cbbb;
      --accent: #0f766e;
      --accent-soft: #cfeeea;
      --warn: #b45309;
      --good: #166534;
      --bad: #b91c1c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, #f9e6c7 0, transparent 24rem),
        linear-gradient(180deg, #f5f0e8 0%, var(--bg) 100%);
      color: var(--ink);
    }
    .wrap {
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px;
    }
    .hero {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      margin-bottom: 20px;
    }
    h1, h2, h3 { margin: 0; font-weight: 700; }
    h1 { font-size: 2rem; }
    h2 { font-size: 1.2rem; margin-bottom: 12px; }
    .sub { color: var(--muted); margin-top: 6px; }
    .grid {
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 16px;
    }
    .card {
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 10px 24px rgba(31, 27, 22, 0.06);
    }
    .span-4 { grid-column: span 4; }
    .span-6 { grid-column: span 6; }
    .span-8 { grid-column: span 8; }
    .span-12 { grid-column: span 12; }
    .stats {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
    }
    .stat {
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255,255,255,0.5);
    }
    .label { color: var(--muted); font-size: 0.86rem; }
    .value { font-size: 1.5rem; margin-top: 6px; font-weight: 700; }
    .pill {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.8rem;
      border: 1px solid var(--line);
      background: #fff;
    }
    .pill.running { color: var(--warn); border-color: #eab308; background: #fef3c7; }
    .pill.completed { color: var(--good); border-color: #86efac; background: #dcfce7; }
    .pill.pending { color: var(--muted); }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.94rem;
    }
    th, td {
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }
    th { color: var(--muted); font-weight: 600; }
    .path {
      font-family: Consolas, monospace;
      font-size: 0.82rem;
      color: var(--muted);
      word-break: break-all;
    }
    .chart {
      margin-top: 12px;
      width: 100%;
      min-height: 260px;
      border-radius: 14px;
      background: linear-gradient(180deg, rgba(15,118,110,0.05), rgba(15,118,110,0.01));
      border: 1px solid var(--line);
      padding: 12px;
    }
    .legend { display: flex; gap: 16px; color: var(--muted); font-size: 0.85rem; margin-top: 8px; }
    .dot { width: 10px; height: 10px; display: inline-block; border-radius: 50%; margin-right: 6px; }
    .muted { color: var(--muted); }
    .mono { font-family: Consolas, monospace; }
    @media (max-width: 960px) {
      .span-4, .span-6, .span-8, .span-12 { grid-column: span 12; }
      .stats { grid-template-columns: repeat(2, 1fr); }
      .hero { display: block; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div>
        <h1>Whisper Pularr Training Dashboard</h1>
        <div class="sub" id="rootLine"></div>
      </div>
      <div class="pill" id="refreshLine"></div>
    </div>
    <div class="grid">
      <section class="card span-4">
        <h2>Pseudo Labeling</h2>
        <div class="stats">
          <div class="stat"><div class="label">Processed</div><div class="value" id="pseudoProcessed">-</div></div>
          <div class="stat"><div class="label">Accepted</div><div class="value" id="pseudoAccepted">-</div></div>
          <div class="stat"><div class="label">Rejected</div><div class="value" id="pseudoRejected">-</div></div>
          <div class="stat"><div class="label">Snapshots</div><div class="value" id="pseudoSnapshots">-</div></div>
        </div>
        <div class="sub" id="pseudoDir"></div>
      </section>

      <section class="card span-8">
        <h2>Current Self-Train Run</h2>
        <div id="currentRun"></div>
      </section>

      <section class="card span-6">
        <h2>Supervised Trials</h2>
        <div id="supervisedTable"></div>
      </section>

      <section class="card span-6">
        <h2>Self-Train Snapshots</h2>
        <div id="selfTrainTable"></div>
      </section>

      <section class="card span-12">
        <h2>WER/CER By Epoch</h2>
        <div class="muted">Validation trend for the currently active self-train run, or the latest completed snapshot if nothing is running.</div>
        <div id="chartWrap"></div>
      </section>
    </div>
  </div>

  <script>
    const refreshSeconds = __REFRESH_SECONDS__;

    function fmt(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return Number(value).toFixed(2) + "%";
    }

    function esc(text) {
      return String(text ?? "").replace(/[&<>"]/g, ch => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;" }[ch]));
    }

    function buildTable(rows, columns) {
      if (!rows.length) return '<div class="muted">No data yet.</div>';
      const head = columns.map(col => `<th>${esc(col.label)}</th>`).join("");
      const body = rows.map(row => (
        `<tr>${columns.map(col => `<td>${col.render(row)}</td>`).join("")}</tr>`
      )).join("");
      return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
    }

    function linePath(points, width, height, minY, maxY) {
      if (!points.length) return "";
      const usableHeight = height - 30;
      const usableWidth = width - 50;
      const stepX = points.length === 1 ? 0 : usableWidth / (points.length - 1);
      return points.map((point, index) => {
        const x = 30 + (index * stepX);
        const y = 10 + usableHeight - (((point.y - minY) / Math.max(maxY - minY, 1)) * usableHeight);
        return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
      }).join(" ");
    }

    function buildChart(epochMetrics) {
      if (!epochMetrics.length) return '<div class="muted">No epoch validation metrics yet.</div>';
      const width = 960;
      const height = 300;
      const pointsWer = epochMetrics.map(item => ({ x: item.epoch, y: item.normalized_wer ?? 0 }));
      const pointsCer = epochMetrics.map(item => ({ x: item.epoch, y: item.normalized_cer ?? 0 }));
      const allY = epochMetrics.flatMap(item => [item.normalized_wer, item.normalized_cer]).filter(v => v !== null && v !== undefined);
      const minY = Math.min(...allY, 0);
      const maxY = Math.max(...allY, 1);
      const werPath = linePath(pointsWer, width, height, minY, maxY);
      const cerPath = linePath(pointsCer, width, height, minY, maxY);
      const labels = epochMetrics.map((item, index) => {
        const usableWidth = width - 50;
        const stepX = epochMetrics.length === 1 ? 0 : usableWidth / (epochMetrics.length - 1);
        const x = 30 + (index * stepX);
        return `<text x="${x}" y="${height - 4}" text-anchor="middle" font-size="11" fill="#74695d">${esc(item.epoch)}</text>`;
      }).join("");
      return `
        <div class="chart">
          <svg viewBox="0 0 ${width} ${height}" width="100%" height="300" preserveAspectRatio="none">
            <line x1="30" y1="10" x2="30" y2="${height - 20}" stroke="#b9ad9d" stroke-width="1"/>
            <line x1="30" y1="${height - 20}" x2="${width - 20}" y2="${height - 20}" stroke="#b9ad9d" stroke-width="1"/>
            <path d="${werPath}" fill="none" stroke="#0f766e" stroke-width="3"/>
            <path d="${cerPath}" fill="none" stroke="#b45309" stroke-width="3"/>
            ${labels}
          </svg>
          <div class="legend">
            <div><span class="dot" style="background:#0f766e"></span>Normalized WER</div>
            <div><span class="dot" style="background:#b45309"></span>Normalized CER</div>
          </div>
        </div>`;
    }

    function render(data) {
      document.getElementById("rootLine").textContent = `Root: ${data.root}`;
      document.getElementById("refreshLine").textContent = `Auto-refresh every ${refreshSeconds}s`;

      const pseudo = data.pseudo_labeling || {};
      const report = pseudo.report || {};
      document.getElementById("pseudoProcessed").textContent = report.processed_samples ?? "-";
      document.getElementById("pseudoAccepted").textContent = report.accepted_samples ?? "-";
      document.getElementById("pseudoRejected").textContent = report.rejected_samples ?? "-";
      document.getElementById("pseudoSnapshots").textContent = pseudo.manifest_count ?? "-";
      document.getElementById("pseudoDir").textContent = pseudo.manifest_dir ? `Snapshots: ${pseudo.manifest_dir}` : "No manifest directory yet.";

      const current = (data.self_train || {}).current_run;
      document.getElementById("currentRun").innerHTML = current ? `
        <div class="stats">
          <div class="stat"><div class="label">Run</div><div class="value" style="font-size:1.1rem">${esc(current.name)}</div></div>
          <div class="stat"><div class="label">Status</div><div class="value" style="font-size:1.1rem"><span class="pill ${esc(current.status)}">${esc(current.status)}</span></div></div>
          <div class="stat"><div class="label">Latest WER</div><div class="value">${fmt(current.latest_normalized_wer ?? current.normalized_wer)}</div></div>
          <div class="stat"><div class="label">Latest CER</div><div class="value">${fmt(current.latest_normalized_cer ?? current.normalized_cer)}</div></div>
        </div>
        <div style="margin-top:12px" class="path">${esc(current.run_dir)}</div>
        <div style="margin-top:10px" class="muted">Latest epoch: ${esc(current.latest_epoch ? current.latest_epoch.epoch : "-")} | Latest raw WER: ${fmt(current.latest_raw_wer ?? current.raw_wer)} | Latest raw CER: ${fmt(current.latest_raw_cer ?? current.raw_cer)}</div>
        <div style="margin-top:8px" class="muted">Best normalized WER/CER: ${fmt(current.normalized_wer)} / ${fmt(current.normalized_cer)}</div>
      ` : '<div class="muted">No self-train run detected yet.</div>';

      document.getElementById("supervisedTable").innerHTML = buildTable(data.supervised_runs || [], [
        { label: "Trial", render: row => esc(row.name) },
        { label: "Status", render: row => `<span class="pill ${esc(row.status)}">${esc(row.status)}</span>` },
        { label: "WER", render: row => fmt(row.normalized_wer) },
        { label: "CER", render: row => fmt(row.normalized_cer) },
        { label: "Best Epoch", render: row => esc(row.best_epoch ?? "-") },
      ]);

      document.getElementById("selfTrainTable").innerHTML = buildTable((data.self_train || {}).runs || [], [
        { label: "Snapshot", render: row => esc(row.name) },
        { label: "Status", render: row => `<span class="pill ${esc(row.status)}">${esc(row.status)}</span>` },
        { label: "WER", render: row => fmt(row.normalized_wer) },
        { label: "CER", render: row => fmt(row.normalized_cer) },
        { label: "Epoch", render: row => esc(row.best_epoch ?? (row.latest_epoch ? row.latest_epoch.epoch : "-")) },
      ]);

      document.getElementById("chartWrap").innerHTML = buildChart(current ? (current.epoch_metrics || []) : []);
    }

    async function fetchData() {
      const response = await fetch("/api/status");
      const data = await response.json();
      render(data);
    }

    fetchData();
    setInterval(fetchData, refreshSeconds * 1000);
  </script>
</body>
</html>
"""


def make_handler(root: Path, refresh_seconds: int):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/status":
                payload = json.dumps(collect_dashboard_data(root), ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if parsed.path == "/" or parsed.path == "/index.html":
                document = HTML.replace("__REFRESH_SECONDS__", str(refresh_seconds)).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(document)))
                self.end_headers()
                self.wfile.write(document)
                return
            if parsed.path == "/healthz":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
                return
            self.send_error(404)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    server = HTTPServer((args.host, args.port), make_handler(root, args.refresh_seconds))
    print(json.dumps({"host": args.host, "port": args.port, "root": str(root)}, sort_keys=True))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
