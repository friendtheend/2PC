#!/usr/bin/env python3
"""Generate a projected LLaMA-2-7B SHAFT vs BriLLMFlow online table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict


COUNTS_32 = {
    "llama_W_QKV": 96,
    "llama_W_O": 32,
    "llama_W_gateup": 64,
    "llama_W_down": 32,
    "llama_QKT": 1024,
    "llama_scoresV": 1024,
}

DISPLAY = {
    "llama_W_QKV": r"W\_Q/K/V",
    "llama_W_O": r"W\_O",
    "llama_W_gateup": r"W\_gate/up",
    "llama_W_down": r"W\_down",
    "llama_QKT": r"Q@K$^T$",
    "llama_scoresV": r"scores@V",
}

ORDER = [
    "llama_W_QKV",
    "llama_W_O",
    "llama_W_gateup",
    "llama_W_down",
    "llama_QKT",
    "llama_scoresV",
]


def get(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return float(value) if str(value).strip() else default


def load_shaft_summary(path: Path, fourier_label: str, d2poly_label: str) -> tuple[Dict[str, str], Dict[str, str]]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    by_label = {row["label"]: row for row in rows}
    missing = [label for label in [fourier_label, d2poly_label] if label not in by_label]
    if missing:
        raise SystemExit(f"{path}: missing SHAFT labels: {', '.join(missing)}")
    return by_label[fourier_label], by_label[d2poly_label]


def project_8_to_32(row: Dict[str, str], field: str) -> float:
    total_key = f"{field}_s" if not field.endswith("_gb") else field
    return get(row, total_key)


def project_time(row: Dict[str, str], component: str) -> float:
    return get(row, f"{component}_s") * 4.0


def project_comm(row: Dict[str, str], component: str) -> float:
    return get(row, f"{component}_gb") * 4.0


def project_total(row: Dict[str, str]) -> float:
    total = get(row, "total_running_time_s")
    embedding = get(row, "embedding_time_s")
    return embedding + (total - embedding) * 4.0


def project_total_comm(row: Dict[str, str]) -> float:
    total = get(row, "total_comm_bytes_gb")
    embedding = get(row, "embedding_comm_bytes_gb")
    return embedding + (total - embedding) * 4.0


def load_brillm(path: Path) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row["label"]
            if label not in COUNTS_32:
                continue
            count = int(float(row.get("count_32layers") or row.get("count") or COUNTS_32[label]))
            if row.get("projected_total_ms", "").strip():
                time_s = float(row["projected_total_ms"]) / 1000.0
                comm_gb = float(row["projected_sent_mb"]) / 1000.0
            else:
                single_s = get(row, "online_s", get(row, "total_s")) 
                single_comm_gb = get(row, "comm_gb", get(row, "sent_mb") / 1000.0)
                time_s = single_s * count
                comm_gb = single_comm_gb * count
            rows[label] = {
                "count": count,
                "time_s": time_s,
                "comm_gb": comm_gb,
                "bw_gbps": comm_gb * 8.0 / time_s if time_s > 0 else math.nan,
            }
    missing = [label for label in ORDER if label not in rows]
    if missing:
        raise SystemExit(f"{path}: missing BriLLMFlow labels: {', '.join(missing)}")
    return rows


def ratio(a: float, b: float) -> float:
    return a / b if b else math.inf


def fmt_ms(s: float) -> str:
    return f"{s * 1000.0:.0f} ms"


def fmt_mb(gb: float) -> str:
    return f"{gb * 1000.0:.0f} MB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shaft-summary", required=True, type=Path)
    parser.add_argument("--brillm-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--fourier-label", default="llama2_fourier")
    parser.add_argument("--d2poly-label", default="llama2_d2poly")
    args = parser.parse_args()

    shaft_fourier, shaft_d2poly = load_shaft_summary(args.shaft_summary, args.fourier_label, args.d2poly_label)
    brillm = load_brillm(args.brillm_csv)

    shaft_total_s = project_total(shaft_fourier)
    shaft_total_gb = project_total_comm(shaft_fourier)
    shaft_linear_s = project_time(shaft_fourier, "matmul_time")
    shaft_linear_gb = project_comm(shaft_fourier, "matmul_comm_bytes")
    shaft_silu_s = project_time(shaft_fourier, "silu_time")
    shaft_silu_gb = project_comm(shaft_fourier, "silu_comm_bytes")
    shaft_nonlin_s = shaft_total_s - shaft_linear_s
    shaft_nonlin_gb = shaft_total_gb - shaft_linear_gb

    brillm_linear_s = sum(row["time_s"] for row in brillm.values())
    brillm_linear_gb = sum(row["comm_gb"] for row in brillm.values())
    d2_total_s = project_total(shaft_d2poly)
    d2_total_gb = project_total_comm(shaft_d2poly)
    d2_linear_s = project_time(shaft_d2poly, "matmul_time")
    d2_linear_gb = project_comm(shaft_d2poly, "matmul_comm_bytes")
    brillm_nonlin_s = d2_total_s - d2_linear_s
    brillm_nonlin_gb = d2_total_gb - d2_linear_gb
    brillm_silu_s = project_time(shaft_d2poly, "silu_time")
    brillm_silu_gb = project_comm(shaft_d2poly, "silu_comm_bytes")
    brillm_total_s = brillm_linear_s + brillm_nonlin_s
    brillm_total_gb = brillm_linear_gb + brillm_nonlin_gb

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": "LLaMA-2-7B, seq=8, 32 layers projected from 8 layers",
        "shaft_total_s": shaft_total_s,
        "shaft_total_comm_gb": shaft_total_gb,
        "shaft_linear_s": shaft_linear_s,
        "shaft_linear_comm_gb": shaft_linear_gb,
        "shaft_nonlin_s": shaft_nonlin_s,
        "shaft_nonlin_comm_gb": shaft_nonlin_gb,
        "shaft_silu_s": shaft_silu_s,
        "shaft_silu_comm_gb": shaft_silu_gb,
        "brillm_linear_s": brillm_linear_s,
        "brillm_linear_comm_gb": brillm_linear_gb,
        "brillm_nonlin_s": brillm_nonlin_s,
        "brillm_nonlin_comm_gb": brillm_nonlin_gb,
        "brillm_silu_s": brillm_silu_s,
        "brillm_silu_comm_gb": brillm_silu_gb,
        "brillm_total_s": brillm_total_s,
        "brillm_total_comm_gb": brillm_total_gb,
        "online_speedup": ratio(shaft_total_s, brillm_total_s),
        "online_comm_less": ratio(shaft_total_gb, brillm_total_gb),
    }
    (args.out_dir / "llama2_hybrid_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.out_dir / "llama2_hybrid_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    with (args.out_dir / "llama2_brillm_perop.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "op", "count", "time_ms", "comm_mb", "bw_gbps"])
        writer.writeheader()
        for label in ORDER:
            row = brillm[label]
            writer.writerow(
                {
                    "label": label,
                    "op": DISPLAY[label].replace("\\_", "_").replace("$", ""),
                    "count": row["count"],
                    "time_ms": row["time_s"] * 1000.0,
                    "comm_mb": row["comm_gb"] * 1000.0,
                    "bw_gbps": row["bw_gbps"],
                }
            )

    latex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{LLaMA-2-7B online phase: projected 32-layer performance from an 8-layer two-party run (seq=8). Vanilla SHAFT uses Fourier-series SiLU; BriLLMFlow applies MixPoly SiLU and replaces linear operators with GPU integer GEMM.}",
        r"\label{tab:llama2_hybrid_comparison}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\footnotesize",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Op} & \textbf{Time (ms)} & \textbf{Comm (MB)} & \textbf{BW (Gbps)} \\",
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{BriLLMFlow Linear (per-operator)}} \\",
    ]
    for label in ORDER:
        row = brillm[label]
        latex.append(
            f"{DISPLAY[label]:<13} & {row['time_s'] * 1000.0:.1f} & {row['comm_gb'] * 1000.0:.1f} & {row['bw_gbps']:.2f} \\\\"
        )
    latex += [
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Aggregate (LLaMA-2-7B, 32 layers)}} \\",
        r"                       & \textbf{SHAFT} & \textbf{BriLLMFlow} & \textbf{Speedup} \\",
        f"Linear time            & {fmt_ms(shaft_linear_s)} & \\textbf{{{fmt_ms(brillm_linear_s)}}} & \\textbf{{{ratio(shaft_linear_s, brillm_linear_s):.2f}}}$\\times$ \\\\",
        f"Linear comm            & {fmt_mb(shaft_linear_gb)} & {fmt_mb(brillm_linear_gb)} & {ratio(shaft_linear_gb, brillm_linear_gb):.2f}$\\times$ less \\\\",
        f"Non-linear time        & {fmt_ms(shaft_nonlin_s)} & \\textbf{{{fmt_ms(brillm_nonlin_s)}}} & {ratio(shaft_nonlin_s, brillm_nonlin_s):.2f}$\\times$ \\\\",
        f"Non-linear comm        & {fmt_mb(shaft_nonlin_gb)} & \\textbf{{{fmt_mb(brillm_nonlin_gb)}}} & {ratio(shaft_nonlin_gb, brillm_nonlin_gb):.2f}$\\times$ less \\\\",
        f"\\quad of which SiLU time & {fmt_ms(shaft_silu_s)} & \\textbf{{{fmt_ms(brillm_silu_s)}}} & \\textbf{{{ratio(shaft_silu_s, brillm_silu_s):.1f}}}$\\times$ \\\\",
        f"\\quad of which SiLU comm & {fmt_mb(shaft_silu_gb)} & \\textbf{{{fmt_mb(brillm_silu_gb)}}} & \\textbf{{{ratio(shaft_silu_gb, brillm_silu_gb):.1f}}}$\\times$ less \\\\",
        r"\midrule",
        f"\\textbf{{Online E2E}}    & {fmt_ms(shaft_total_s)} & \\textbf{{{fmt_ms(brillm_total_s)}}} & \\textbf{{{ratio(shaft_total_s, brillm_total_s):.2f}}}$\\times$ \\\\",
        f"\\textbf{{Online comm}}   & {fmt_mb(shaft_total_gb)} & \\textbf{{{fmt_mb(brillm_total_gb)}}} & \\textbf{{{ratio(shaft_total_gb, brillm_total_gb):.2f}}}$\\times$ less \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    tex = "\n".join(latex)
    (args.out_dir / "llama2_table.tex").write_text(tex, encoding="utf-8")
    (args.out_dir / "table5.tex").write_text(tex, encoding="utf-8")
    print(tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
