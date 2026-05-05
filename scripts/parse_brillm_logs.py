#!/usr/bin/env python3
"""Parse BriLLMFlow BMT offline/online logs into CSV."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def last(pattern: str, text: str) -> Optional[Tuple[str, ...]]:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    value = matches[-1]
    return value if isinstance(value, tuple) else (value,)


def parse_log_spec(spec: str) -> Tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label, Path(path)
    path = Path(spec)
    return path.stem, path


def parse_offline(label: str, path: Path, text: str) -> Dict[str, object]:
    done = last(r"^\[Party 0\] Done:\s*([0-9.]+)\s+s", text)
    timing = last(
        r"^\[Party 0\] Time:\s*PRG=([0-9.]+)s,\s*LocalMM=([0-9.]+)s,\s*Gilboa=([0-9.]+)s,\s*File=([0-9.]+)s",
        text,
    )
    comm = last(
        r"^\[Party 0\] Communication:\s*Sent=([0-9.]+)\s+MB,\s*Recv=([0-9.]+)\s+MB,\s*Total=([0-9.]+)\s+MB",
        text,
    )
    tput = last(r"^\[Party 0\] Throughput:\s*([0-9.]+)\s+C elements/s", text)
    ots = last(r"^\[Party 0\] OT calls:\s*([0-9]+),\s*Total OTs:\s*([0-9]+)", text)
    shape = last(
        r"Matrix:\s*\[([0-9]+).([0-9]+)\]\s*.?\s*\[([0-9]+).([0-9]+)\]\s*=\s*\[([0-9]+).([0-9]+)\]",
        text,
    )

    if done is None:
        raise ValueError(f"{path}: could not parse Party 0 Done line")
    total_s = float(done[0])
    sent_mb = float(comm[0]) if comm else 0.0
    recv_mb = float(comm[1]) if comm else 0.0

    row: Dict[str, object] = {
        "label": label,
        "kind": "offline",
        "total_s": total_s,
        "online_s": "",
        "offline_s": total_s,
        "sent_mb": sent_mb,
        "recv_mb": recv_mb,
        "wire_tb": sent_mb / 1e6,
        "comm_gb": sent_mb / 1000.0,
        "bw_gbps": sent_mb * 8e6 / total_s / 1e9 if total_s > 0 else "",
        "tput_cps": float(tput[0]) if tput else "",
        "prg_s": float(timing[0]) if timing else "",
        "localmm_s": float(timing[1]) if timing else "",
        "gilboa_s": float(timing[2]) if timing else "",
        "file_s": float(timing[3]) if timing else "",
        "ot_calls": int(ots[0]) if ots else "",
        "total_ots": int(ots[1]) if ots else "",
        "M": int(shape[4]) if shape else "",
        "K": int(shape[1]) if shape else "",
        "N": int(shape[5]) if shape else "",
        "source_log": str(path),
    }
    return row


def parse_online(label: str, path: Path, text: str) -> Dict[str, object]:
    iterations = last(r"Iterations:\s*([0-9]+)", text)
    total = last(r"Total time:\s*([0-9.]+)\s+s", text)
    prg = last(r"PRG time:\s*([0-9.]+)\s+ms / iter", text)
    comm_time = last(r"Comm time:\s*([0-9.]+)\s+ms / iter", text)
    gemm = last(r"GEMM time:\s*([0-9.]+)\s+ms / iter", text)
    comm_bytes = last(
        r"^\[Party 0\] Communication per iter:\s*Sent=([0-9.]+)\s+MB,\s*Recv=([0-9.]+)\s+MB,\s*Total=([0-9.]+)\s+MB",
        text,
    )
    shape = last(
        r"Matrix:\s*\[([0-9]+).([0-9]+)\]\s*.?\s*\[([0-9]+).([0-9]+)\]\s*=\s*\[([0-9]+).([0-9]+)\]",
        text,
    )

    iter_rows = re.findall(
        r"Iter\s+[0-9]+:\s*total=([0-9.]+)\s+ms,\s*PRG=([0-9.]+)\s+ms,\s*comm=([0-9.]+)\s+ms,\s*GEMM=([0-9.]+)\s+ms",
        text,
    )
    if iter_rows:
        iters = len(iter_rows)
        online_s = sum(float(r[0]) for r in iter_rows) / iters / 1000.0
        total_s_all = online_s * iters
        prg_s = sum(float(r[1]) for r in iter_rows) / iters / 1000.0
        comm_s = sum(float(r[2]) for r in iter_rows) / iters / 1000.0
        gemm_s = sum(float(r[3]) for r in iter_rows) / iters / 1000.0
    elif total is not None and iterations is not None:
        iters = int(iterations[0])
        total_s_all = float(total[0])
        online_s = total_s_all / iters if iters > 0 else total_s_all
        prg_s = float(prg[0]) / 1000.0 if prg else ""
        comm_s = float(comm_time[0]) / 1000.0 if comm_time else ""
        gemm_s = float(gemm[0]) / 1000.0 if gemm else ""
    else:
        raise ValueError(f"{path}: could not parse online timing")

    sent_mb = float(comm_bytes[0]) if comm_bytes else 0.0
    recv_mb = float(comm_bytes[1]) if comm_bytes else 0.0

    row: Dict[str, object] = {
        "label": label,
        "kind": "online",
        "total_s": online_s,
        "online_s": online_s,
        "offline_s": "",
        "sent_mb": sent_mb,
        "recv_mb": recv_mb,
        "wire_tb": sent_mb / 1e6,
        "comm_gb": sent_mb / 1000.0,
        "bw_gbps": sent_mb * 8e6 / online_s / 1e9 if online_s > 0 else "",
        "tput_cps": "",
        "prg_s": prg_s,
        "localmm_s": "",
        "gilboa_s": "",
        "file_s": "",
        "ot_calls": "",
        "total_ots": "",
        "M": int(shape[4]) if shape else "",
        "K": int(shape[1]) if shape else "",
        "N": int(shape[5]) if shape else "",
        "iterations": iters,
        "total_s_all_iterations": total_s_all,
        "comm_s": comm_s,
        "gemm_s": gemm_s,
        "source_log": str(path),
    }
    return row


def detect_kind(text: str) -> str:
    if "GPU Matrix Beaver Online" in text or re.search(r"Iter\s+[0-9]+:\s*total=", text):
        return "online"
    return "offline"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["auto", "offline", "online"], default="auto")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("logs", nargs="+", help="Either label=/path/to/log or /path/to/log")
    args = parser.parse_args()

    rows: List[Dict[str, object]] = []
    for spec in args.logs:
        label, path = parse_log_spec(spec)
        text = path.read_text(encoding="utf-8", errors="ignore")
        kind = detect_kind(text) if args.kind == "auto" else args.kind
        if kind == "online":
            rows.append(parse_online(label, path, text))
        else:
            rows.append(parse_offline(label, path, text))

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
