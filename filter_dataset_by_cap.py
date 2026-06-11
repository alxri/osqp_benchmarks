#!/usr/bin/env python3
import os
import sys
import json
import glob
import shutil


def filter_dataset(src='dataset', dst='dataset_filtered', cap=32768):
    os.makedirs(dst, exist_ok=True)
    copied = 0
    skipped = 0

    for json_file in glob.glob(os.path.join(src, '**', '*.json'), recursive=True):
        try:
            with open(json_file, 'r') as f:
                meta = json.load(f)
        except Exception:
            skipped += 1
            continue

        n = int(meta.get('n', 0))
        m = int(meta.get('m', 0))

        rel = os.path.relpath(json_file, src)
        dst_json = os.path.join(dst, rel)
        dst_dir = os.path.dirname(dst_json)
        npz_file = os.path.splitext(json_file)[0] + '.npz'
        dst_npz = os.path.splitext(dst_json)[0] + '.npz'

        if n <= cap and m <= cap and os.path.exists(npz_file):
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(json_file, dst_json)
            shutil.copy2(npz_file, dst_npz)
            copied += 1
        else:
            skipped += 1

    print(f"Copied {copied} problems to {dst}. Skipped {skipped}.")


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else 'dataset'
    dst = sys.argv[2] if len(sys.argv) > 2 else 'dataset_filtered'
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else 32768
    filter_dataset(src, dst, cap)
