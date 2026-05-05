#!/usr/bin/env python3
"""Generate a GPT-2 SHAFT vs BriLLMFlow online table.

The BriLLMFlow CSV may contain either one row per logical GPT-2 row with a
`count` column, or one measured occurrence per label. If `count` is missing,
the GPT-2-base, seq=128, 12-layer counts below are applied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Optional


GPT2_COUNTS = {
    "gpt2_W_QKV": 36,
    "gpt2_W_O": 12,
    "gpt2_W_1": 12,
    "gpt2_W_2": 12,
    "gpt2_QKT": 144,
    "gpt2_scoresV": 144,
}

DISPLAY = {
    "gpt2_W_QKV": r"W\_Q/K/V",
    "gpt2_W_O": r"W\_O",
    "gpt2_W_1": r"W\_1",
    "gpt2_W_2": r"W\_2",
    "gpt2_QKT": r"Q@K$^T$",
    "gpt2_scoresV": r"scores@V",
}

ORDER = [
    "gpt2_W_QKV",
    "gpt2_W_O",
    "gpt2_W_1",
    "gpt2_W_2",
    "gpt2_QKT",
    "gpt2_scoresV",
]


def floats(pattern: str, text: str) -> list[float]:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return [float(x) for x in matches]


def max_float(pattern: str, text: str) -> Optional[float]:
    values = floats(pattern, text)
    return max(values) if values else None


def parse_shaft(path: Path) -> Dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    fields = {}
    for name in [
        "total_running_time",
        "matmul_time",
        "gelu_time",
        "total_comm_bytes",
        "matmul_comm_bytes",
        "gelu_comm_bytes",
    ]:
        unit = "sec" if name.endswith("_time") else "GB"
        value = max_float(rf"^{re.escape(name)}:\s*([0-9.eE+-]+)\s+{unit}\b", text)
        if value is not None:
            fields[name] = value

    if "total_running_time" not in fields:
        value = max_float(r"running time:\s*([0-9.eE+-]+)s", text)
        if value is not None:
            fields["total_running_time"] = value
    if "total_comm_bytes" not in fields:
        value = max_float(r"comm byte:\s*([0-9.eE+-]+)\s+GB", text)
        if value is not None:
            fields["total_comm_bytes"] = value

    missing = [
        name
        for name in ["total_running_time", "matmul_time", "gelu_time", "total_comm_bytes", "matmul_comm_bytes", "gelu_comm_bytes"]
        if name not in fields
    ]
    if missing:
        raise SystemExit(f"{path}: missing SHAFT report_cost fields: {', '.join(missing)}")
    return fields


def get_float(row: Dict[str, str], names: Iterable[str], default: Optional[float] = None) -> Optional[float]:
    for name in names:
        value = row.get(name, "")
        if str(value).strip() != "":
            return float(value)
    return default


def load_brillm(path: Path) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row["label"]
            if label not in GPT2_COUNTS:
                continue

            count = int(float(row.get("count") or GPT2_COUNTS[label]))
            single_s = get_float(row, ["single_online_s"])
            if single_s is None:
                # `parse_brillm_logs.py` emits a single occurrence in online_s.
                single_s = get_float(row, ["online_s", "total_s", "time_s"], 0.0)
                already_counted = "count" in row and row.get("count", "").strip() != ""
            else:
                already_counted = False

            single_comm_gb = get_float(row, ["single_comm_gb"])
            if single_comm_gb is None:
                single_comm_gb = get_float(row, ["comm_gb", "wire_gb"], None)
            if single_comm_gb is None:
                wire_tb = get_float(row, ["wire_tb", "wire_tb_avg"], None)
                single_comm_gb = (wire_tb * 1000.0) if wire_tb is not None else None
            if single_comm_gb is None:
                sent_mb = get_float(row, ["sent_mb", "sent_mb_avg"], 0.0)
                single_comm_gb = sent_mb / 1000.0

            if already_counted:
                time_s = get_float(row, ["online_s", "total_s", "time_s"], 0.0)
                comm_gb = get_float(row, ["comm_gb"], single_comm_gb)
            else:
                time_s = single_s * count
                comm_gb = single_comm_gb * count

            rows[label] = {
                "count": count,
                "time_s": time_s,
                "comm_gb": comm_gb,
                "bw_gbps": (comm_gb * 8.0 / time_s) if time_s > 0 else math.nan,
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
    parser.add_argument("--shaft-log", required=True, type=Path)
    parser.add_argument("--brillm-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-title", default="GPT2-base, 12 layers")
    parser.add_argument("--brillm-gelu-ms", type=float, default=68.0)
    parser.add_argument("--brillm-gelu-mb", type=float, default=70.0)
    args = parser.parse_args()

    shaft = parse_shaft(args.shaft_log)
    brillm = load_brillm(args.brillm_csv)

    shaft_total_s = shaft["total_running_time"]
    shaft_linear_s = shaft["matmul_time"]
    shaft_gelu_s = shaft["gelu_time"]
    shaft_total_gb = shaft["total_comm_bytes"]
    shaft_linear_gb = shaft["matmul_comm_bytes"]
    shaft_gelu_gb = shaft["gelu_comm_bytes"]

    shaft_nonlin_s = shaft_total_s - shaft_linear_s
    shaft_nonlin_gb = shaft_total_gb - shaft_linear_gb

    brillm_linear_s = sum(row["time_s"] for row in brillm.values())
    brillm_linear_gb = sum(row["comm_gb"] for row in brillm.values())
    brillm_gelu_s = args.brillm_gelu_ms / 1000.0
    brillm_gelu_gb = args.brillm_gelu_mb / 1000.0
    brillm_nonlin_s = shaft_nonlin_s - shaft_gelu_s + brillm_gelu_s
    brillm_nonlin_gb = shaft_nonlin_gb - shaft_gelu_gb + brillm_gelu_gb
    brillm_total_s = brillm_linear_s + brillm_nonlin_s
    brillm_total_gb = brillm_linear_gb + brillm_nonlin_gb

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": args.model_title,
        "shaft_total_s": shaft_total_s,
        "shaft_total_comm_gb": shaft_total_gb,
        "shaft_linear_s": shaft_linear_s,
        "shaft_linear_comm_gb": shaft_linear_gb,
        "shaft_nonlin_s": shaft_nonlin_s,
        "shaft_nonlin_comm_gb": shaft_nonlin_gb,
        "shaft_gelu_s": shaft_gelu_s,
        "shaft_gelu_comm_gb": shaft_gelu_gb,
        "brillm_linear_s": brillm_linear_s,
        "brillm_linear_comm_gb": brillm_linear_gb,
        "brillm_nonlin_s": brillm_nonlin_s,
        "brillm_nonlin_comm_gb": brillm_nonlin_gb,
        "brillm_gelu_s": brillm_gelu_s,
        "brillm_gelu_comm_gb": brillm_gelu_gb,
        "brillm_total_s": brillm_total_s,
        "brillm_total_comm_gb": brillm_total_gb,
        "online_speedup": ratio(shaft_total_s, brillm_total_s),
        "online_comm_less": ratio(shaft_total_gb, brillm_total_gb),
    }
    (args.out_dir / "gpt2_hybrid_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (args.out_dir / "gpt2_hybrid_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    with (args.out_dir / "gpt2_brillm_perop.csv").open("w", newline="", encoding="utf-8") as f:
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

    latex = []
    latex += [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{GPT2-base online phase comparison with SHAFT (sequence length 128, 12 layers). SHAFT uses its standard GPT2 private generation path; BriLLMFlow replaces linear operators with GPU integer GEMM and uses MixPoly GELU.}",
        r"\label{tab:gpt2_shaft_comparison}",
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
            f"{DISPLAY[label]:<11} & {row['time_s'] * 1000.0:.1f} & {row['comm_gb'] * 1000.0:.1f} & {row['bw_gbps']:.2f} \\\\"
        )
    latex += [
        r"\midrule",
        rf"\multicolumn{{4}}{{l}}{{\textit{{Aggregate ({args.model_title})}}}} \\",
        r"                       & \textbf{SHAFT} & \textbf{BriLLMFlow} & \textbf{Speedup} \\",
        f"Linear time            & {fmt_ms(shaft_linear_s)} & \\textbf{{{fmt_ms(brillm_linear_s)}}} & \\textbf{{{ratio(shaft_linear_s, brillm_linear_s):.2f}}}$\\times$ \\\\",
        f"Linear comm            & {fmt_mb(shaft_linear_gb)} & {fmt_mb(brillm_linear_gb)} & {ratio(shaft_linear_gb, brillm_linear_gb):.2f}$\\times$ less \\\\",
        f"Non-linear time        & {fmt_ms(shaft_nonlin_s)} & \\textbf{{{fmt_ms(brillm_nonlin_s)}}} & {ratio(shaft_nonlin_s, brillm_nonlin_s):.2f}$\\times$ \\\\",
        f"Non-linear comm        & {fmt_mb(shaft_nonlin_gb)} & \\textbf{{{fmt_mb(brillm_nonlin_gb)}}} & {ratio(shaft_nonlin_gb, brillm_nonlin_gb):.2f}$\\times$ less \\\\",
        f"\\quad of which GELU time & {fmt_ms(shaft_gelu_s)} & \\textbf{{{fmt_ms(brillm_gelu_s)}}} & \\textbf{{{ratio(shaft_gelu_s, brillm_gelu_s):.1f}}}$\\times$ \\\\",
        f"\\quad of which GELU comm & {fmt_mb(shaft_gelu_gb)} & \\textbf{{{fmt_mb(brillm_gelu_gb)}}} & \\textbf{{{ratio(shaft_gelu_gb, brillm_gelu_gb):.1f}}}$\\times$ less \\\\",
        r"\midrule",
        f"\\textbf{{Online E2E}}    & {fmt_ms(shaft_total_s)} & \\textbf{{{fmt_ms(brillm_total_s)}}} & \\textbf{{{ratio(shaft_total_s, brillm_total_s):.2f}}}$\\times$ \\\\",
        f"\\textbf{{Online comm}}   & {fmt_mb(shaft_total_gb)} & \\textbf{{{fmt_mb(brillm_total_gb)}}} & \\textbf{{{ratio(shaft_total_gb, brillm_total_gb):.2f}}}$\\times$ less \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    (args.out_dir / "gpt2_table.tex").write_text("\n".join(latex), encoding="utf-8")
    print("\n".join(latex))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
