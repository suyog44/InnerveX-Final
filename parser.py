# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import argparse
import base64
import csv
import io
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================
# Data model
# ============================

@dataclass
class DetRow:
    timestamp: datetime
    label: str
    confidence: float
    threat_score: float
    camouflage: int
    x1: float
    y1: float
    x2: float
    y2: float


# ============================
# Parsing helpers
# ============================

def parse_dt(s: str) -> datetime:
    s = s.strip()
    return datetime.fromisoformat(s)


def read_csv(path: Path) -> List[DetRow]:
    rows: List[DetRow] = []

    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        required = {
            "timestamp", "label", "confidence",
            "threat_score", "camouflage",
            "x1", "y1", "x2", "y2"
        }

        if not required.issubset(set(r.fieldnames or [])):
            raise ValueError(f"CSV columns mismatch. Found: {r.fieldnames}")

        for row in r:
            try:
                rows.append(
                    DetRow(
                        timestamp=parse_dt(row["timestamp"]),
                        label=row["label"],
                        confidence=float(row["confidence"]),
                        threat_score=float(row["threat_score"]),
                        camouflage=int(row["camouflage"]),
                        x1=float(row["x1"]),
                        y1=float(row["y1"]),
                        x2=float(row["x2"]),
                        y2=float(row["y2"]),
                    )
                )
            except Exception:
                continue

    return rows


# ============================
# Time bucketing
# ============================

def floor_time(dt: datetime, span: str) -> datetime:
    if span == "minute":
        return dt.replace(second=0, microsecond=0)
    if span == "5min":
        return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)
    if span == "15min":
        return dt.replace(minute=(dt.minute // 15) * 15, second=0, microsecond=0)
    if span == "hour":
        return dt.replace(minute=0, second=0, microsecond=0)
    if span == "day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    raise ValueError(span)


# ============================
# Plot helpers
# ============================

def save_fig(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    return buf.getvalue()


def b64_png(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


# ============================
# Plots
# ============================

def make_histogram(rows: List[DetRow], span: str) -> Tuple[bytes, Dict[datetime, int]]:
    counts: Dict[datetime, int] = {}
    for r in rows:
        b = floor_time(r.timestamp, span)
        counts[b] = counts.get(b, 0) + 1

    buckets = sorted(counts)
    values = [counts[b] for b in buckets]

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.bar(range(len(values)), values, color="#4C78A8")
    ax.set_title("Detections over time")
    ax.set_ylabel("Count")

    step = max(1, len(buckets) // 12)
    ax.set_xticks(range(0, len(buckets), step))
    ax.set_xticklabels(
        [buckets[i].strftime("%Y-%m-%d %H:%M") for i in range(0, len(buckets), step)],
        rotation=45,
        ha="right",
    )

    return save_fig(fig), counts


def make_pie(rows: List[DetRow]) -> Tuple[bytes, Dict[str, int]]:
    counts: Dict[str, int] = {}
    for r in rows:
        lab = r.label or "unknown"
        counts[lab] = counts.get(lab, 0) + 1

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.pie(counts.values(), labels=counts.keys(), autopct="%1.1f%%", startangle=140)
    ax.set_title("Detections by label")

    return save_fig(fig), counts


def make_top_threats(rows: List[DetRow], topk: int) -> bytes:
    rows = sorted(rows, key=lambda r: r.threat_score, reverse=True)[:topk]
    rows = sorted(rows, key=lambda r: r.timestamp)

    times = [r.timestamp for r in rows]
    scores = [r.threat_score for r in rows]
    colors = ["#ff4d4d" if r.camouflage else "#ffb020" for r in rows]

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.scatter(times, scores, c=colors)
    ax.plot(times, scores, alpha=0.4)
    ax.set_ylim(0, 1.05)
    ax.set_title("Top threat detections")
    ax.set_ylabel("Threat score")

    fig.autofmt_xdate()
    return save_fig(fig)


# ============================
# HTML report
# ============================

def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&#39;")
    )


def build_html_report(
    rows: List[DetRow],
    span: str,
    hist_b64: str,
    pie_b64: str,
    top_b64: str,
    label_counts: Dict[str, int],
    out_path: Path,
):
    total = len(rows)
    avg_threat = sum(r.threat_score for r in rows) / total if total else 0
    camo = sum(1 for r in rows if r.camouflage)

    recent_rows = sorted(rows, key=lambda r: r.timestamp, reverse=True)[:25]

    label_rows = "".join(
        f"<tr><td>{html_escape(k)}</td><td>{v}</td></tr>"
        for k, v in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    )

    recent_html = "".join(
        f"<tr><td>{r.timestamp.isoformat()}</td><td>{html_escape(r.label)}</td>"
        f"<td>{r.confidence:.2f}</td><td>{r.threat_score:.2f}</td>"
        f"<td>{'YES' if r.camouflage else 'NO'}</td></tr>"
        for r in recent_rows
    )

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>SPECTRA Detection Report</title>
<style>
body {{
  background:#0b0f14;
  color:#e6edf3;
  font-family:Inter,Arial,sans-serif;
  padding:24px;
}}
h1 {{ margin-bottom:4px; }}
.card {{
  background:#141a22;
  border:1px solid #243040;
  border-radius:14px;
  padding:16px;
  margin-bottom:16px;
}}
.grid {{
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:16px;
}}
.kpis {{
  display:grid;
  grid-template-columns:repeat(4,1fr);
  gap:12px;
}}
.kpi {{
  background:#0e1520;
  padding:14px;
  border-radius:12px;
}}
img {{
  width:100%;
  border-radius:10px;
  border:1px solid #243040;
}}
table {{
  width:100%;
  border-collapse:collapse;
}}
th,td {{
  padding:8px;
  border-bottom:1px solid #243040;
}}
.footer {{
  text-align:center;
  margin-top:24px;
  color:#9aa4b2;
  font-size:12px;
}}
</style>
</head>

<body>

<h1>SPECTRA  Detection Intelligence Report</h1>

<div class="card kpis">
  <div class="kpi">Total<br><b>{total}</b></div>
  <div class="kpi">Camouflage<br><b>{camo}</b></div>
  <div class="kpi">Avg Threat<br><b>{avg_threat:.3f}</b></div>
  <div class="kpi">Bucket<br><b>{span}</b></div>
</div>

<div class="grid">
  <div class="card"><h2>Detections over time</h2><img src="data:image/png;base64,{hist_b64}"/></div>
  <div class="card"><h2>By label</h2><img src="data:image/png;base64,{pie_b64}"/></div>
</div>

<div class="card">
  <h2>Top threats</h2>
  <img src="data:image/png;base64,{top_b64}"/>
</div>

<div class="grid">
  <div class="card">
    <h2>Label counts</h2>
    <table><tr><th>Label</th><th>Count</th></tr>{label_rows}</table>
  </div>
  <div class="card">
    <h2>Recent detections</h2>
    <table>
      <tr><th>Time</th><th>Label</th><th>Conf</th><th>Threat</th><th>Camo</th></tr>
      {recent_html}
    </table>
  </div>
</div>

<div class="footer">
Generated {datetime.now(timezone.utc).isoformat()} UTC
</div>

</body>
</html>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


# ============================
# Main
# ============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/detections.csv")
    ap.add_argument("--out", default="data/report.html")
    ap.add_argument("--span", default="15min",
                    choices=["minute","5min","15min","hour","day"])
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    rows = read_csv(Path(args.csv))
    hist_png, _ = make_histogram(rows, args.span)
    pie_png, labels = make_pie(rows)
    top_png = make_top_threats(rows, args.topk)

    build_html_report(
        rows,
        args.span,
        b64_png(hist_png),
        b64_png(pie_png),
        b64_png(top_png),
        labels,
        Path(args.out),
    )

    print(f"[OK] Report generated: {args.out}")


if __name__ == "__main__":
    main()
