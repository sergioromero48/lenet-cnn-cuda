#!/usr/bin/env python3
import struct, os, sys

def main():
    path = "model/model.dat"
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        return

    # read entire file as signed 8-bit weights
    with open(path, "rb") as f:
        data = f.read()
    # unpack all bytes as signed 8-bit ints
    count = len(data)
    weights = struct.unpack(f"{count}b", data)
    # write all weights to a text file
    out_path = "weights.txt"
    with open(out_path, "w") as out_f:
        for i, w in enumerate(weights):
            out_f.write(f"{i}: {w}\n")
    print(f"Wrote {count} weights to {out_path}")

if __name__ == "__main__":
    main()