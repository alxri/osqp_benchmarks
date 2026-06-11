#!/usr/bin/env python3
import os
import sys
import json
import glob

def main(path='dataset', cap=32768):
    path = sys.argv[1] if len(sys.argv) > 1 else path
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else cap

    exceed = []
    max_n = 0
    max_m = 0
    total = 0

    for json_file in glob.glob(os.path.join(path, '**', '*.json'), recursive=True):
        total += 1
        try:
            with open(json_file, 'r') as f:
                meta = json.load(f)

            n = int(meta.get('n', 0))
            m = int(meta.get('m', 0))
            orig = meta.get('original_dim', None)

            max_n = max(max_n, n)
            max_m = max(max_m, m)

            if n > cap or m > cap:
                exceed.append((json_file, n, m, orig))

        except Exception as e:
            print(f"Error reading {json_file}: {e}", file=sys.stderr)

    if exceed:
        print(f"Found {len(exceed)} files exceeding cap {cap}:")
        for f, n, m, orig in exceed:
            print(f"{f}: n={n}, m={m}, original_dim={orig}")
    else:
        print(f"No files exceed cap {cap}.")

    print(f"Checked {total} json files. max n={max_n}, max m={max_m}.")


if __name__ == '__main__':
    main()
