#!/usr/bin/env python3
"""Parse [OP_PROFILE] lines from SHAFT logs and export raw/aggregated CSV."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


LINE_RE = re.compile(
    r"^\[OP_PROFILE\]\s+"
    r"idx=(?P<idx>\d+)\s+"
    r"node=(?P<node>\S+)\s+"
    r"module=(?P<module>\S+)\s+"
    r"input_shape=(?P<input_shape>.+?)\s+"
    r"output_shape=(?P<output_shape>.+?)\s+"
    r"time_ms=(?P<time_ms>[0-9.eE+-]+)\s+"
    r"comm_mb=(?P<comm_mb>[0-9.eE+-]+)\s+"
    r"rounds=(?P<rounds>[0-9.eE+-]+)\s*$"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--raw-out", required=True, type=Path)
    parser.add_argument("--agg-out", required=True, type=Path)
    args = parser.parse_args()

    raw_rows = []
    for line in args.log.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        d = m.groupdict()
        raw_rows.append(
            {
                "idx": int(d["idx"]),
                "node": d["node"],
                "module": d["module"],
                "input_shape": d["input_shape"],
                "output_shape": d["output_shape"],
                "time_ms": float(d["time_ms"]),
                "comm_mb": float(d["comm_mb"]),
                "rounds": float(d["rounds"]),
            }
        )

    args.raw_out.parent.mkdir(parents=True, exist_ok=True)
    with args.raw_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["idx", "node", "module", "input_shape", "output_shape", "time_ms", "comm_mb", "rounds"]
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    agg = defaultdict(lambda: {"count": 0, "time_ms_sum": 0.0, "comm_mb_sum": 0.0, "rounds_sum": 0.0})
    for row in raw_rows:
        key = (row["node"], row["module"], row["input_shape"], row["output_shape"])
        agg[key]["count"] += 1
        agg[key]["time_ms_sum"] += row["time_ms"]
        agg[key]["comm_mb_sum"] += row["comm_mb"]
        agg[key]["rounds_sum"] += row["rounds"]

    agg_rows = []
    for (node, module, input_shape, output_shape), st in sorted(agg.items(), key=lambda x: x[1]["time_ms_sum"], reverse=True):
        count = st["count"]
        agg_rows.append(
            {
                "node": node,
                "module": module,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "count": count,
                "time_ms_sum": st["time_ms_sum"],
                "time_ms_avg": st["time_ms_sum"] / count if count else 0.0,
                "comm_mb_sum": st["comm_mb_sum"],
                "comm_mb_avg": st["comm_mb_sum"] / count if count else 0.0,
                "rounds_sum": st["rounds_sum"],
                "rounds_avg": st["rounds_sum"] / count if count else 0.0,
            }
        )

    with args.agg_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "node",
                "module",
                "input_shape",
                "output_shape",
                "count",
                "time_ms_sum",
                "time_ms_avg",
                "comm_mb_sum",
                "comm_mb_avg",
                "rounds_sum",
                "rounds_avg",
            ],
        )
        writer.writeheader()
        writer.writerows(agg_rows)

    print(f"Parsed {len(raw_rows)} OP_PROFILE rows from {args.log}")
    print(f"Raw CSV: {args.raw_out}")
    print(f"Agg CSV: {args.agg_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
