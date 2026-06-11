'''
Run a single, specific benchmark instance.
Usage example:
python run_single_problem.py --problem Lasso --n 1632 --density 15 --rho 0 --solver indirect
'''

import os
import sys

# --- DYNAMIC LD_LIBRARY_PATH INJECTION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_lib_dir = os.path.join(script_dir, '.venv_bench', 'lib')
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')

if venv_lib_dir not in current_ld_path:
    print(f"[*] Auto-configuring LD_LIBRARY_PATH with: {venv_lib_dir}")
    os.environ['LD_LIBRARY_PATH'] = f"{venv_lib_dir}:{current_ld_path}".strip(':')
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import numpy as np
import scipy.sparse as spa
import solvers.solvers as s
from solvers.solvers import SOLVER_MAP
from benchmark_problems.example import LoadedExample, RAPLMeter

def main():
    parser = argparse.ArgumentParser(description='Run a specific problem instance.')
    
    # Target Parameters
    parser.add_argument('--problem', required=True, type=str,
                        choices=['Random QP', 'Eq QP', 'Portfolio', 'Lasso', 'SVM', 'Huber', 'Control'],
                        help='Exact problem class name')
    parser.add_argument('--n', required=True, type=int, help='Dimension (n) of the problem')
    parser.add_argument('--density', required=True, type=int, help='Density parameter')
    parser.add_argument('--rho', required=True, type=int, choices=[0, 1], help='Adaptive rho setting (0 or 1)')
    parser.add_argument('--inst', type=int, default=0, help='Instance number (default: 0)')
    
    # Solver Parameters
    parser.add_argument('--solver', default='both', choices=['both', 'direct', 'indirect'])
    parser.add_argument('--dataset-dir', type=str, default='dataset')
    parser.add_argument('--high_accuracy', action='store_true', default=False)
    
    args = parser.parse_args()

    # Determine Solvers
    if args.solver == 'both':
        solvers = [s.OSQP_builtin_direct, s.OSQP_mkl_indirect]
    elif args.solver == 'direct':
        solvers = [s.OSQP_builtin_direct]
    else:
        solvers = [s.OSQP_mkl_indirect]

    # Map human-readable names to folder names
    PROBLEMS_MAP = {
        'Random QP': 'random_qp', 'Eq QP': 'eq_qp', 'Portfolio': 'portfolio',
        'Lasso': 'lasso', 'SVM': 'svm', 'Huber': 'huber', 'Control': 'control'
    }

    # Locate the target file
    prob_dir_name = PROBLEMS_MAP[args.problem]
    npz_file = os.path.join(args.dataset_dir, prob_dir_name, f"density_{args.density}", f"n{args.n}_inst{args.inst}.npz")

    print("======================================================")
    print(f"Target Problem: {args.problem} | n={args.n} | Density={args.density} | Rho={args.rho}")
    print(f"File Path:      {npz_file}")
    print("======================================================\n")

    if not os.path.exists(npz_file):
        print(f"[!] CRITICAL: Dataset file missing or not generated: {npz_file}")
        sys.exit(1)

    # Load Matrix Data
    example_instance = LoadedExample(args.problem, npz_file)
    P = example_instance.qp_problem['P']
    A = example_instance.qp_problem['A']
    q_val = example_instance.qp_problem['q']
    l_bound = example_instance.qp_problem['l']
    u_bound = example_instance.qp_problem['u']
    
    print(f"[Matrix Details] n: {example_instance.qp_problem['n']}, m: {example_instance.qp_problem['m']}, nnz(P): {P.nnz}, nnz(A): {A.nnz}\n")

    # Override Settings
    if args.high_accuracy:
        for key in solvers:
            s.settings[key]['eps_abs'] = 1e-05
            s.settings[key]['eps_rel'] = 1e-05
            
    for key in solvers:
        s.settings[key]['max_iter'] = 40000

    # Run Solvers
    for solver_name in solvers:
        print(f"--- Running {solver_name} ---")
        settings = s.settings[solver_name].copy()
        settings['adaptive_rho'] = args.rho
        settings['verbose'] = True  # Force output to terminal to watch convergence
        
        solver_obj = SOLVER_MAP[solver_name](settings)
        meter = RAPLMeter()
        
        start_energy = meter.read()
        results = solver_obj.solve(example_instance)
        end_energy = meter.read()
        
        # Calculate Math Fallback Residuals
        pri_res, dua_res = np.nan, np.nan
        if results.x is not None and results.y is not None and len(results.x) > 0:
            Ax = A.dot(results.x)
            z_val = np.clip(Ax, l_bound, u_bound)
            pri_res = np.max(np.abs(Ax - z_val))
            
            Px = P.dot(results.x)
            ATy = A.T.dot(results.y)
            dua_res = np.max(np.abs(Px + q_val + ATy))

        # Format Energy
        energy_str = "N/A"
        if not np.isnan(start_energy) and not np.isnan(end_energy):
            energy_str = f"{(end_energy - start_energy):.4f} Joules"

        print(f"Status:       {results.status}")
        print(f"Iterations:   {results.niter}")
        print(f"Total Time:   {results.run_time:.4f} s (Setup: {getattr(results, 'setup_time', 0.0):.4f} s, Solve: {getattr(results, 'solve_time', 0.0):.4f} s)")
        print(f"Energy:       {energy_str}")
        print(f"Objective:    {results.obj_val:.4f}")
        print(f"Primal Res:   {pri_res:.2e}")
        print(f"Dual Res:     {dua_res:.2e}\n")

if __name__ == "__main__":
    main()