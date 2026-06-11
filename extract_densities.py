#!/usr/bin/env python3
import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Calculate nnz/col densities directly from results.csv.")
    parser.add_argument('--input', type=str, default='results/benchmark_problems_osqp_only/OSQP_builtin_direct/results.csv', help='Input CSV file')
    parser.add_argument('--output', type=str, default='results_actual_densities.csv', help='Output CSV file')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    # Load the CSV
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)

    # Check if necessary columns exist
    required_cols = ['n', 'm', 'nnz_A', 'nnz_P']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' missing from the CSV.")
            return

    # --- CALCULATE DENSITIES ---
    # P is an n x n matrix. Columns = n.
    df['actual_nnz_per_col_P'] = (df['nnz_P'] / df['n']).round(2)

    # A is an m x n matrix (m rows, n columns). Columns = n.
    df['actual_nnz_per_col_A'] = (df['nnz_A'] / df['n']).round(2)
    
    # Keep only the columns relevant to the structural analysis
    cols_to_keep = []
    for optional_col in ['class', 'solver', 'density']:
        if optional_col in df.columns:
            cols_to_keep.append(optional_col)
            
    # Reordered to place nnz per col P and A right next to each other
    cols_to_keep += [
        'n', 'm', 
        'nnz_P', 'nnz_A', 
        'actual_nnz_per_col_P', 'actual_nnz_per_col_A'
    ]

    # Drop duplicates so you get one row per matrix size/configuration, 
    # instead of duplicating it for rho=0 and rho=1
    out_df = df[cols_to_keep].drop_duplicates()

    # Save to the new CSV
    out_df.to_csv(args.output, index=False)
    print(f"Success! Analyzed {len(out_df)} unique matrix configurations.")
    print(f"Saved detailed density analysis to: {args.output}")

if __name__ == "__main__":
    main()