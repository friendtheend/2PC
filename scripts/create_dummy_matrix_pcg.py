#!/usr/bin/env python3
"""Create dummy Matrix Beaver files for gpu_matrix_beaver_online.cu.

The online benchmark reads only the file header and the C-share array. For
timing the online path, the actual C values are irrelevant, so this script
creates zero C shares with the correct header and dimensions.
"""

import argparse
import os
import struct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, help="Output prefix; files are <prefix>0.bin and <prefix>1.bin")
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--bits", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.prefix)), exist_ok=True)
    nC = args.M * args.N
    for party in (0, 1):
        path = f"{args.prefix}{party}.bin"
        seed_hi = 0x123456789ABCDEF0 + party
        seed_lo = 0xFEDCBA9876543210 ^ party
        with open(path, "wb") as f:
            f.write(struct.pack("<iiiiQQi", party, args.M, args.K, args.N, seed_hi, seed_lo, args.bits))
            f.write(b"\x00" * (nC * 8))
        print(f"{path} {os.path.getsize(path)}")


if __name__ == "__main__":
    main()
