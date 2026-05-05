#!/usr/bin/env python3
"""Parse SHAFT/CrypTen report_cost logs into one CSV.

Usage:
  scripts/parse_shaft_report.py --out results/shaft_summary.csv \
    vit_d2poly=results/vit_d2poly.log \
    llama_fourier=results/llama_fourier.log
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TIME_FIELDS = [
    "total_running_time",
    "embedding_time",
    "matmul_time",
    "softmax_time",
    "gelu_time",
    "silu_time",
    "layernorm_time",
    "tanh_time",
    "conv_time",
    "other_time",
    "total_comm_time",
    "embedding_comm_time",
    "matmul_comm_time",
    "softmax_comm_time",
    "gelu_comm_time",
    "silu_comm_time",
    "layernorm_comm_time",
    "tanh_comm_time",
    "conv_comm_time",
    "other_comm_time",
]

COMM_FIELDS = [
    "total_comm_bytes",
    "embedding_comm_bytes",
    "matmul_comm_bytes",
    "softmax_comm_bytes",
    "gelu_comm_bytes",
    "silu_comm_bytes",
    "layernorm_comm_bytes",
    "tanh_comm_bytes",
    "conv_comm_bytes",
    "other_comm_bytes",
]

ROUND_FIELDS = [
    "total_comm_rounds",
    "embedding_comm_rounds",
    "matmul_comm_rounds",
    "softmax_comm_rounds",
    "gelu_comm_rounds",
    "silu_comm_rounds",
    "layernorm_comm_rounds",
    "tanh_comm_rounds",
    "conv_comm_rounds",
    "other_comm_rounds",
]


def parse_spec(spec: str) -> Tuple[str, Path]:
    if "=" in spec:
        label, raw_path = spec.split("=", 1)
        return label, Path(raw_path)
    path = Path(spec)
    return path.stem, path


def all_floats(pattern: str, text: str) -> List[float]:
    return [float(x) for x in re.findall(pattern, text, flags=re.MULTILINE)]


def max_float(pattern: str, text: str) -> Optional[float]:
    values = all_floats(pattern, text)
    return max(values) if values else None


def parse_one(label: str, path: Path) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    row: Dict[str, object] = {"label": label, "source_log": str(path)}

    comp = max_float(r"comp time:\s*([0-9.eE+-]+)s", text)
    if comp is not None:
        row["comp_time_s"] = comp

    compact_total = max_float(r"running time:\s*([0-9.eE+-]+)s", text)
    compact_comm = max_float(r"comm byte:\s*([0-9.eE+-]+)\s+GB", text)
    bench_latency = max_float(r"BENCH_RESULT[^\n]*latency_sec=([0-9.eE+-]+)", text)
    bench_comm = max_float(r"BENCH_RESULT[^\n]*comm_gb=([0-9.eE+-]+)", text)
    if compact_total is not None:
        row["compact_total_s"] = compact_total
    if compact_comm is not None:
        row["compact_comm_gb"] = compact_comm
    if bench_latency is not None:
        row["bench_latency_s"] = bench_latency
    if bench_comm is not None:
        row["bench_comm_gb"] = bench_comm

    for field in TIME_FIELDS:
        value = max_float(rf"^{re.escape(field)}:\s*([0-9.eE+-]+)\s+sec\b", text)
        if value is not None:
            row[f"{field}_s"] = value
    for field in COMM_FIELDS:
        value = max_float(rf"^{re.escape(field)}:\s*([0-9.eE+-]+)\s+GB\b", text)
        if value is not None:
            row[f"{field}_gb"] = value
    for field in ROUND_FIELDS:
        value = max_float(rf"^{re.escape(field)}:\s*([0-9.eE+-]+)\s*$", text)
        if value is not None:
            row[field] = int(value)

    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("logs", nargs="+", help="Either label=/path/to/log or /path/to/log")
    args = parser.parse_args()

    rows = [parse_one(*parse_spec(spec)) for spec in args.logs]
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
