#!/usr/bin/env python3
"""Extract OSQP problem sizes from generated dataset JSON metadata.

Outputs CSV with columns:
  path,problem,density,original_dim,n,m,nnzA,nnzP

Usage:
  python benchmarks/osqp_benchmarks/extract_problem_sizes.py dataset/ > sizes.csv
  python benchmarks/osqp_benchmarks/extract_problem_sizes.py dataset/ sizes.csv
"""
import sys
import os
import glob
import json
import csv


def extract(src='dataset', out=None):
    files = sorted(glob.glob(os.path.join(src, '**', '*.json'), recursive=True))
    rows = []

    for jf in files:
        try:
            with open(jf, 'r') as f:
                meta = json.load(f)
        except Exception:
            continue

        row = {
            'path': jf,
            'problem': meta.get('problem', ''),
            'density': meta.get('density', ''),
            'original_dim': meta.get('original_dim', ''),
            'n': meta.get('n', ''),
            'm': meta.get('m', ''),
            'nnzA': meta.get('nnzA', ''),
            'nnzP': meta.get('nnzP', ''),
        }
        rows.append(row)

    fieldnames = ['path', 'problem', 'density', 'original_dim', 'n', 'm', 'nnzA', 'nnzP']

    if out is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    else:
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        with open(out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else 'dataset'
    out = sys.argv[2] if len(sys.argv) > 2 else None
    extract(src, out)
