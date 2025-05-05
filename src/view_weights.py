#!/usr/bin/env python3
import struct, os, sys

def main():
    path = "model/model.dat"
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        return

    # read only the first 100 bytes (weights)
    with open(path, "rb") as f:
        data = f.read(100)
    if len(data) < 100:
        print(f"File too short: only {len(data)} bytes", file=sys.stderr)
        return

    # unpack as signed 8-bit ints
    weights = struct.unpack("100b", data)
    for i, w in enumerate(weights):
        print(f"{i:3d}: {w}")

if __name__ == "__main__":
    main()