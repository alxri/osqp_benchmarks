import os
import sys

# --- DYNAMIC LD_LIBRARY_PATH INJECTION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_lib_dir = os.path.join(script_dir, '.venv_bench', 'lib')
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')

if venv_lib_dir not in current_ld_path:
    print(f"[*] Auto-configuring LD_LIBRARY_PATH with: {venv_lib_dir}")
    os.environ['LD_LIBRARY_PATH'] = f"{venv_lib_dir}:{current_ld_path}".strip(':')
    # Re-execute the current Python script with the updated environment
    os.execv(sys.executable, [sys.executable] + sys.argv)

# 1. FAIL-SAFE: Ensure the library path is set before importing qpfpga
if "QPFPGA_LIBRARY" not in os.environ:
    # Look for the .so file in the expected build directory
    expected_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cpp", "build", "libqpfpga.so"))
    if os.path.exists(expected_path):
        os.environ["QPFPGA_LIBRARY"] = expected_path
    else:
        print(f"ERROR: Could not find libqpfpga.so at {expected_path}")
        print("Please compile the backend or set QPFPGA_LIBRARY manually.")
        sys.exit(1)

import qpfpga  # This triggers the registration of cp.QPFPGA

import re
import argparse
import glob
import pandas as pd
from benchmark_problems.example import Example
import solvers.solvers as s
from utils.benchmark import compute_stats_info

parser = argparse.ArgumentParser(description='Dynamic Dataset Benchmark Runner (FPGA only)')
parser.add_argument('--high_accuracy', help='Test with high accuracy', default=False, action='store_true')
parser.add_argument('--verbose', help='Verbose solvers', default=False, action='store_true')
parser.add_argument('--parallel', help='Parallel solution', default=False, action='store_true')
parser.add_argument('--no-cvxpy', help='Skip CVXPY model construction', default=False, action='store_true')
parser.add_argument('--debug', help='Print benchmark step timings', default=False, action='store_true')
parser.add_argument('--exclude-problems', nargs='*', default=[], choices=['Random QP', 'Eq QP', 'Portfolio', 'Lasso', 'SVM', 'Huber', 'Control'], help='Problem types to skip')
parser.add_argument('--dataset-dir', type=str, default='dataset', help='Path to the pre-generated dataset directory')

args = parser.parse_args()
high_accuracy = args.high_accuracy
verbose = args.verbose
parallel = args.parallel
build_cvxpy = not args.no_cvxpy
debug = args.debug
exclude_problems = set(args.exclude_problems)
dataset_dir = args.dataset_dir

n_repeats = 1 # Number of solve calls per problem instance

print('high_accuracy', high_accuracy)
print('verbose', verbose)
print('parallel', parallel)
print('build_cvxpy', build_cvxpy)
print('debug', debug)
print('exclude_problems', sorted(exclude_problems))
print('dataset_dir', dataset_dir)

solvers = [s.QPFPGA]

OUTPUT_FOLDER = 'benchmark_problems_fpga_only'

if high_accuracy:
    for key in solvers:
        s.settings[key]['eps_abs'] = 1e-05
        s.settings[key]['eps_rel'] = 1e-05
        s.settings[key]['high_accuracy'] = True

for key in solvers:
    s.settings[key]['admm_max_iter'] = 40000
    s.settings[key]['measure_energy'] = True

if verbose:
    for key in solvers:
        s.settings[key]['verbose'] = True

if debug:
    for key in solvers:
        s.settings[key]['debug'] = True

# Map human-readable names to folder names used by dataset generator
PROBLEMS_MAP = {
    'Random QP': 'random_qp',
    'Eq QP': 'eq_qp',
    'Portfolio': 'portfolio',
    'Lasso': 'lasso',
    'SVM': 'svm',
    'Huber': 'huber',
    'Control': 'control'
}

problems = [p for p in PROBLEMS_MAP.keys() if p not in exclude_problems]

if not problems:
    raise ValueError('No problems selected after applying --exclude-problems')

# --- DYNAMIC DATASET DISCOVERY ENGINE ---
print("\nScanning dataset directory for available instances...")

for problem in problems:
    folder_name = PROBLEMS_MAP[problem]
    prob_path = os.path.join(dataset_dir, folder_name)
    
    if not os.path.exists(prob_path):
        print(f"Skipping {problem}: Folder '{prob_path}' not found.")
        continue
        
    # Discover all density directories (e.g., density_1, density_3)
    density_dirs = glob.glob(os.path.join(prob_path, "density_*"))
    
    for d_dir in sorted(density_dirs):
        # Extract density number from folder name
        density_match = re.search(r'density_(\d+)', d_dir)
        if not density_match:
            continue
        density = int(density_match.group(1))
        
        # Discover all .npz files in this density sub-directory
        npz_files = glob.glob(os.path.join(d_dir, "*.npz"))
        if not npz_files:
            continue
            
        # Parse out available dimensions (n) and count instances per dimension
        dims_dict = {}  # { dimension_int: max_instance_number }
        for f in npz_files:
            filename = os.path.basename(f)
            match = re.match(r'n(\d+)_inst(\d+)\.npz', filename)

            if match:
                dim_val = int(match.group(1))
                inst_val = int(match.group(2))

                if dim_val not in dims_dict:
                    dims_dict[dim_val] = []

                dims_dict[dim_val].append(inst_val)

        for dim in dims_dict:
            dims_dict[dim] = sorted(dims_dict[dim])
                
        if not dims_dict:
            continue
            
        # Extract dimensions in sorted order
        sorted_dims = sorted(dims_dict.keys())
        
        if debug:
            print(
                f"[debug] Found {problem} "
                f"(Density {density}): "
                f"Dims={sorted_dims}, "
                f"Instance map={dims_dict}"
            )
        example = Example(
            problem,
            sorted_dims,
            solvers,
            s.settings,
            OUTPUT_FOLDER,
            instance_map=dims_dict,
            debug=debug,
            nnz_per_col=density,
            build_cvxpy=build_cvxpy,
            dataset_dir=dataset_dir,
            n_repeats=n_repeats
        )
                          
        # Execute solution pass
        example.solve(parallel=parallel)

print("\nAll solver loops complete.")
print("Merging density-specific CSVs into master full.csv files for analysis...")

valid_problems_for_stats = set()

for solver in solvers:
    for problem in problems:
        # Path where example.py saved the results
        res_path = os.path.join('.', 'results', OUTPUT_FOLDER, solver, problem)
        if os.path.exists(res_path):
            # Find all density-specific CSVs
            density_files = glob.glob(os.path.join(res_path, 'full_density_*.csv'))
            if density_files:
                # Read and combine them into a single DataFrame
                combined_df = pd.concat([pd.read_csv(f) for f in density_files])
                # Export the master file that compute_stats_info is looking for
                combined_df.to_csv(os.path.join(res_path, 'full.csv'), index=False)
                # Keep track of problems that successfully generated data
                valid_problems_for_stats.add(problem)

valid_problems_list = [p for p in problems if p in valid_problems_for_stats]

if not valid_problems_list:
    print("No valid results found to compute stats. Exiting.")
else:
    print(f"Computing final benchmark stats for: {valid_problems_list}")
    compute_stats_info(solvers, OUTPUT_FOLDER,
                       problems=valid_problems_list,
                       high_accuracy=high_accuracy)