#!/usr/bin/env python3
"""
Generate sparsity pattern plots (spy plots) for P and A matrices 
across different problem classes and densities.
"""

import os
import re
import glob
import argparse
import numpy as np
import scipy.sparse as spa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name).strip("_")

def load_matrices(npz_file):
    """Load P and A from the compressed NPZ file."""
    data = np.load(npz_file)
    P = spa.csc_matrix((data['P_data'], data['P_indices'], data['P_indptr']), shape=data['P_shape'])
    A = spa.csc_matrix((data['A_data'], data['A_indices'], data['A_indptr']), shape=data['A_shape'])
    return P, A

def plot_sparsity(P, A, problem_name, density, out_file):
    """Plot the sparsity patterns side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    max_dim = max(P.shape[0], P.shape[1], A.shape[0], A.shape[1])
    marker_size = max(0.01, min(2.0, 500.0 / max_dim))

    # Statistics
    p_nnz_per_col = P.nnz / P.shape[1] if P.shape[1] > 0 else 0.0
    a_nnz_per_col = A.nnz / A.shape[1] if A.shape[1] > 0 else 0.0

    p_density_pct = 100.0 * P.nnz / (P.shape[0] * P.shape[1])
    a_density_pct = 100.0 * A.nnz / (A.shape[0] * A.shape[1])

    # --- P ---
    axes[0].spy(P, markersize=marker_size, color='#1f77b4', alpha=0.8)
    axes[0].set_title(
        f"$\\bf{{P\\ Matrix}}$\n"
        f"{P.shape[0]:,} × {P.shape[1]:,}\n"
        f"NNZ: {P.nnz:,}\n"
        f"{p_nnz_per_col:.1f} nnz/col ({p_density_pct:.3f}%)",
        fontsize=12
    )
    axes[0].set_xlabel("Columns")
    axes[0].set_ylabel("Rows")

    # --- A ---
    axes[1].spy(A, markersize=marker_size, color='#ff7f0e', alpha=0.8)
    axes[1].set_title(
        f"$\\bf{{A\\ Matrix}}$\n"
        f"{A.shape[0]:,} × {A.shape[1]:,}\n"
        f"NNZ: {A.nnz:,}\n"
        f"{a_nnz_per_col:.1f} nnz/col ({a_density_pct:.3f}%)",
        fontsize=12
    )
    axes[1].set_xlabel("Columns")
    axes[1].set_ylabel("Rows")

    plt.suptitle(
        f"Sparsity Pattern: {problem_name.replace('_', ' ').title()}",
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate sparsity pattern graphs from NPZ dataset.')
    parser.add_argument('--dataset-dir', type=str, default='dataset', help='Path to the dataset directory')
    parser.add_argument('--output-dir', type=str, default='sparsity_plots', help='Output directory for the images')
    args = parser.parse_args()

    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory '{args.dataset_dir}' not found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Group files by Problem and Density
    problem_folders = glob.glob(os.path.join(args.dataset_dir, "*"))
    
    plotted_count = 0

    for prob_path in problem_folders:
        if not os.path.isdir(prob_path):
            continue
            
        problem_name = os.path.basename(prob_path)
        density_folders = glob.glob(os.path.join(prob_path, "density_*"))
        
        for den_path in density_folders:
            density_match = re.search(r'density_(\d+)', os.path.basename(den_path))
            if not density_match:
                continue
            density = int(density_match.group(1))
            
            # Find all npz files in this density folder
            npz_files = glob.glob(os.path.join(den_path, "*.npz"))
            if not npz_files:
                continue
                
            # Parse the dimensions (n) from filenames to find the SMALLEST one
            # Plotting the smallest one gives the clearest visual of the actual pattern!
            file_dim_map = {}
            for f in npz_files:
                match = re.match(r'n(\d+)_inst\d+\.npz', os.path.basename(f))
                if match:
                    file_dim_map[f] = int(match.group(1))
                    
            if not file_dim_map:
                continue
                
            # Select the file with the minimum 'n'
            target_file = min(file_dim_map, key=file_dim_map.get)
            
            print(f"Plotting {problem_name} (Density {density}) using {os.path.basename(target_file)}...")
            
            try:
                P, A = load_matrices(target_file)
                
                out_filename = f"sparsity_{problem_name}_density_{density}.png"
                out_filepath = os.path.join(args.output_dir, out_filename)
                
                plot_sparsity(P, A, problem_name, density, out_filepath)
                plotted_count += 1
            except Exception as e:
                print(f"  [!] Failed to plot {target_file}: {e}")

    print(f"\nSuccess! Generated {plotted_count} sparsity pattern graphs in '{args.output_dir}'.")

if __name__ == "__main__":
    main()