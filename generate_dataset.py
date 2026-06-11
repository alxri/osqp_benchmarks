import os
import json
import argparse
import hashlib
import numpy as np
import scipy.sparse as spa

from problem_classes.random_qp import RandomQPExample
from problem_classes.eq_qp import EqQPExample
from problem_classes.portfolio import PortfolioExample
from problem_classes.lasso import LassoExample
from problem_classes.svm import SVMExample
from problem_classes.huber import HuberExample
from problem_classes.control import ControlExample


PROBLEMS_MAP = {
    'Random QP': RandomQPExample,
    'Eq QP': EqQPExample,
    'Portfolio': PortfolioExample,
    'Lasso': LassoExample,
    'SVM': SVMExample,
    'Huber': HuberExample,
    'Control': ControlExample
}


def stable_seed(problem_name, d, density, inst):
    """
    Deterministic seed that is stable across Python sessions and machines.
    """
    return int(
        hashlib.sha256(
            f"{problem_name}_{d}_{density}_{inst}".encode()
        ).hexdigest()[:8],
        16
    )


def generate_datasets():
    parser = argparse.ArgumentParser(description='Offline Dataset Generator')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='dataset',
        help='Output directory'
    )

    args = parser.parse_args()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Number of instances per dimension and density
    n_instances = 1
    n_dim = 30
    max_osqp_dim = 32768

    # --- Elevated feature densities for ML problems ---
    PROBLEM_DENSITIES = {
        # 'Random QP': [1, 3, 6],
        # 'Eq QP':     [1, 3, 6],
        # 'Control':   [1, 3, 6, 10],
        # 'Portfolio': [1, 3, 6, 10],
        'Lasso':     [6, 15, 40],
        'SVM':       [6, 15, 40],
        'Huber':     [6, 15, 40] 
    }

    problem_input_caps = {
        'Random QP': max_osqp_dim,
        'Eq QP': max_osqp_dim,
        'Portfolio': int((max_osqp_dim - 1) / 101),
        'Lasso': int(max_osqp_dim / 102),
        'SVM': int(max_osqp_dim / 200),
        'Huber': int(max_osqp_dim / 301),
        'Control': int(max_osqp_dim / 27),
    }

    problem_start_dims = {
        'Random QP': 1024,
        'Eq QP': 1024,
        'Portfolio': 10,  
        'Lasso': 10,       
        'SVM': 10,         
        'Huber': 10,       
        'Control': 38      
    }

    problem_dimensions = {}
    for problem_name, cap in problem_input_caps.items():
        start_dim = problem_start_dims[problem_name]
        actual_start = min(start_dim, cap)
        raw_dims = np.geomspace(actual_start, cap, n_dim)
        problem_dimensions[problem_name] = np.unique(np.round(raw_dims).astype(int)).tolist()

    # Enforce strict hardware limits
    for problem_name in problem_dimensions:
        cap = problem_input_caps[problem_name]
        valid_dims = [d for d in problem_dimensions[problem_name] if d < cap]
        valid_dims.append(cap)
        problem_dimensions[problem_name] = sorted(list(set(valid_dims)))

    for problem_name, problem_class in PROBLEMS_MAP.items():
        
        if problem_name not in PROBLEM_DENSITIES:
            print(f"Skipping {problem_name} entirely (Commented out in config).")
            continue

        prob_dir_name = problem_name.lower().replace(' ', '_')
        dims = problem_dimensions[problem_name]
        
        target_densities = PROBLEM_DENSITIES[problem_name]

        for density in target_densities:

            valid_dims = [d for d in dims if d >= density]

            if not valid_dims:
                print(f"Skipping {problem_name} density={density} (no valid dimensions)")
                continue

            density_dir = os.path.join(out_dir, prob_dir_name, f"density_{density}")
            os.makedirs(density_dir, exist_ok=True)

            for d in valid_dims:
                if density > d:
                    continue

                for inst in range(n_instances):

                    filename_base = os.path.join(density_dir, f"n{d}_inst{inst}")
                    npz_file = filename_base + ".npz"
                    json_file = filename_base + ".json"

                    if os.path.exists(npz_file) and os.path.exists(json_file):
                        print(f"Skipping {npz_file} (already exists)")
                        continue

                    print(f"Generating {problem_name} - dim={d}, density={density}, instance={inst}")

                    seed = stable_seed(problem_name, d, density, inst)

                    try:
                        problem = problem_class(
                            d,
                            seed=seed,
                            nnz_per_col=density,
                            build_cvxpy=False
                        )
                    except Exception as e:
                        print(f"Error generating {problem_name} (dim={d}, density={density}): {e}")
                        continue

                    qp = problem.qp_problem
                    P = qp['P'].tocsc()
                    A = qp['A'].tocsc()

                    np.savez_compressed(
                        npz_file,
                        P_data=P.data,
                        P_indices=P.indices,
                        P_indptr=P.indptr,
                        P_shape=P.shape,
                        A_data=A.data,
                        A_indices=A.indices,
                        A_indptr=A.indptr,
                        A_shape=A.shape,
                        q=qp['q'],
                        l=qp['l'],
                        u=qp['u']
                    )

                    meta = {
                        "problem": problem_name,
                        "density": density,
                        "instance": inst,
                        "n": int(qp['n']),
                        "m": int(qp['m']),
                        "nnzA": int(A.nnz),
                        "nnzP": int(P.nnz),
                        "original_dim": int(d)
                    }

                    with open(json_file, 'w') as f:
                        json.dump(meta, f, indent=4)

    print("\nDataset generation complete.")


if __name__ == "__main__":
    generate_datasets()