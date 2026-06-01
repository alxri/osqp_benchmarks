'''
Run benchmark problems with OSQP only, comparing:
    1) builtin + direct linear solver
    2) mkl + indirect linear solver
'''

from benchmark_problems.example import Example
import solvers.solvers as s
from utils.general import gen_int_log_space
from utils.benchmark import compute_stats_info
import argparse


parser = argparse.ArgumentParser(description='Benchmark Problems Runner (OSQP only)')
parser.add_argument('--high_accuracy', help='Test with high accuracy', default=False,
                    action='store_true')
parser.add_argument('--verbose', help='Verbose solvers', default=False,
                    action='store_true')
parser.add_argument('--parallel', help='Parallel solution', default=False,
                    action='store_true')
parser.add_argument('--no-cvxpy', help='Skip CVXPY model construction', default=False,
                    action='store_true')
parser.add_argument('--solver', help='Solver selection', default='both',
                    choices=['both', 'direct', 'indirect'])
parser.add_argument('--debug', help='Print benchmark step timings', default=False,
                    action='store_true')
parser.add_argument('--exclude-problems', nargs='*', default=[],
                    choices=['Random QP', 'Eq QP', 'Portfolio', 'Lasso', 'SVM', 'Huber', 'Control'],
                    help='Problem types to skip')
parser.add_argument('--min-nnz-per-col', type=int, default=1,
                    help='Minimum nonzeros per column for sparse random matrices')
parser.add_argument('--max-nnz-per-col', type=int, default=5,
                    help='Maximum nonzeros per column for sparse random matrices')
args = parser.parse_args()
high_accuracy = args.high_accuracy
verbose = args.verbose
parallel = args.parallel
build_cvxpy = not args.no_cvxpy
solver_mode = args.solver
debug = args.debug
exclude_problems = set(args.exclude_problems)
min_nnz_per_col = args.min_nnz_per_col
max_nnz_per_col = args.max_nnz_per_col

print('high_accuracy', high_accuracy)
print('verbose', verbose)
print('parallel', parallel)
print('build_cvxpy', build_cvxpy)
print('solver', solver_mode)
print('debug', debug)
print('exclude_problems', sorted(exclude_problems))
print('min_nnz_per_col', min_nnz_per_col)
print('max_nnz_per_col', max_nnz_per_col)

if solver_mode == 'both':
    solvers = [s.OSQP_builtin_direct, s.OSQP_mkl_indirect]
elif solver_mode == 'direct':
    solvers = [s.OSQP_builtin_direct]
else:
    solvers = [s.OSQP_mkl_indirect]

OUTPUT_FOLDER = 'benchmark_problems_osqp_only'

if high_accuracy:
    # Tighten the indirect configuration while preserving algebra/solver_type
    for key in solvers:
        s.settings[key]['eps_abs'] = 1e-05
        s.settings[key]['eps_rel'] = 1e-05
        s.settings[key]['high_accuracy'] = True

if verbose:
    for key in solvers:
        s.settings[key]['verbose'] = True

if debug:
    for key in solvers:
        s.settings[key]['debug'] = True


# Number of instances per different dimension
n_instances = 3 # How many instances to solve per dimension, per problem class. Set to 1 for quick testing, or higher for more robust stats.
n_dim = 30 # Number of different dimensions to test, logarithmically spaced between 10 and max_dim
max_total_variables = 30000

all_problems = [
    'Random QP',
    'Eq QP',
    'Portfolio',
    'Lasso',
    'SVM',
    'Huber',
    'Control',
]


# Raw input dimensions are capped so the expanded QP size stays below
# max_total_variables for each problem family.
problem_input_caps = {
    'Random QP': max_total_variables,
    'Eq QP': max_total_variables,
    # Portfolio uses k factors and expands to roughly 101*k variables.
    'Portfolio': max_total_variables,
    # Lasso and SVM expand to roughly 102*n and 101*n variables respectively.
    'Lasso': max_total_variables,
    'SVM': max_total_variables,
    # Huber expands to roughly 301*n variables.
    'Huber': max_total_variables,
    # Control expands to roughly 16*n variables.
    'Control': max_total_variables,
}


def size_cap(problem_name, min_dim):
    # gen_int_log_space adds min_dim back after rounding, so subtract it here
    # to make the generated sizes top out at the per-problem input cap.
    cap = problem_input_caps[problem_name]
    if cap < min_dim:
        raise ValueError(
            f'{problem_name} cap {cap} is smaller than the minimum raw dimension {min_dim}'
        )
    return cap - min_dim + 1


problems = [problem for problem in all_problems if problem not in exclude_problems]

if not problems:
    raise ValueError('No problems selected after applying --exclude-problems')

problem_dimensions = {'Random QP': gen_int_log_space(10, size_cap('Random QP', 10), n_dim),
                                            'Eq QP': gen_int_log_space(10, size_cap('Eq QP', 10), n_dim),
                                            'Portfolio': gen_int_log_space(5, size_cap('Portfolio', 5), n_dim),
                                            'Lasso': gen_int_log_space(10, size_cap('Lasso', 10), n_dim),
                                            'SVM': gen_int_log_space(10, size_cap('SVM', 10), n_dim),
                                            'Huber': gen_int_log_space(10, size_cap('Huber', 10), n_dim),
                                            'Control': gen_int_log_space(10, size_cap('Control', 10), n_dim)}

problem_parallel = {'Random QP': parallel,
                    'Eq QP': parallel,
                                        'Portfolio': parallel,
                    'Lasso': parallel,
                    'SVM': parallel,
                    'Huber': parallel,
                    'Control': parallel}

for problem in problems:
    example = Example(problem,
                      problem_dimensions[problem],
                      solvers,
                      s.settings,
                      OUTPUT_FOLDER,
                      n_instances,
                      debug=debug,
                      min_nnz_per_col=min_nnz_per_col,
                      max_nnz_per_col=max_nnz_per_col,
                      build_cvxpy=build_cvxpy)
    example.solve(parallel=problem_parallel[problem])

compute_stats_info(solvers, OUTPUT_FOLDER,
                   problems=problems,
                   high_accuracy=high_accuracy)
