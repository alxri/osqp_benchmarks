from solvers.osqp import OSQPSolver
from solvers.qpfpga import QPFPGA_Solver

# Optional solver backends: keep imports lazy/fault-tolerant so
# OSQP-only runs do not require proprietary/extra dependencies.
try:
       from solvers.ecos import ECOSSolver
except Exception:
       ECOSSolver = None

try:
       from solvers.gurobi import GUROBISolver
except Exception:
       GUROBISolver = None

try:
       from solvers.mosek import MOSEKSolver
except Exception:
       MOSEKSolver = None

try:
       from solvers.qpoases import qpOASESSolver
except Exception:
       qpOASESSolver = None

ECOS = 'ECOS'
ECOS_high = ECOS + "_high"
GUROBI = 'GUROBI'
GUROBI_high = GUROBI + "_high"
OSQP = 'OSQP'
OSQP_high = OSQP + '_high'
OSQP_polish = OSQP + '_polish'
OSQP_polish_high = OSQP_polish + '_high'
OSQP_builtin_direct = 'OSQP_builtin_direct'
OSQP_mkl_indirect = 'OSQP_mkl_indirect'
QPFPGA = 'QPFPGA'
MOSEK = 'MOSEK'
MOSEK_high = MOSEK + "_high"
qpOASES = 'qpOASES'

# solvers = [ECOSSolver, GUROBISolver, MOSEKSolver, OSQPSolver]
# SOLVER_MAP = {solver.name(): solver for solver in solvers}

SOLVER_MAP = {
       OSQP: OSQPSolver,
       OSQP_high: OSQPSolver,
       OSQP_polish: OSQPSolver,
       OSQP_polish_high: OSQPSolver,
       OSQP_builtin_direct: OSQPSolver,
       OSQP_mkl_indirect: OSQPSolver,
       QPFPGA: QPFPGA_Solver,
}

if GUROBISolver is not None:
       SOLVER_MAP[GUROBI] = GUROBISolver
       SOLVER_MAP[GUROBI_high] = GUROBISolver
if MOSEKSolver is not None:
       SOLVER_MAP[MOSEK] = MOSEKSolver
       SOLVER_MAP[MOSEK_high] = MOSEKSolver
if ECOSSolver is not None:
       SOLVER_MAP[ECOS] = ECOSSolver
       SOLVER_MAP[ECOS_high] = ECOSSolver
if qpOASESSolver is not None:
       SOLVER_MAP[qpOASES] = qpOASESSolver

time_limit = 1000. # Seconds
eps_low = 1e-03
eps_high = 1e-05

# Solver settings
settings = {
    OSQP: {'eps_abs': eps_low,
           'eps_rel': eps_low,
           'polish': False,
           'max_iter': int(1e09),
           'eps_prim_inf': 1e-15,  # Disable infeas check
           'eps_dual_inf': 1e-15,
    },
    OSQP_high: {'eps_abs': eps_high,
                'eps_rel': eps_high,
                'polish': False,
                'max_iter': int(1e09),
                'eps_prim_inf': 1e-15,  # Disable infeas check
                'eps_dual_inf': 1e-15
    },
    OSQP_polish: {'eps_abs': eps_low,
                  'eps_rel': eps_low,
                  'polish': True,
                  'max_iter': int(1e09),
                  'eps_prim_inf': 1e-15,  # Disable infeas check
                  'eps_dual_inf': 1e-15
    },
    OSQP_polish_high: {'eps_abs': eps_high,
                       'eps_rel': eps_high,
                       'polish': True,
                       'max_iter': int(1e09),
                       'eps_prim_inf': 1e-15,  # Disable infeas check
                       'eps_dual_inf': 1e-15
    },
       OSQP_builtin_direct: {'eps_abs': eps_low,
                                            'eps_rel': eps_low,
                                            'warm_start': False,
                                            'polish': False,
                                            'max_iter': 4000,
                                            'eps_prim_inf': 1e-15,  # Disable infeas check
                                            'eps_dual_inf': 1e-15,
                                            'algebra': 'builtin',
                                            'solver_type': 'direct'
                                            
       },
       OSQP_mkl_indirect: {'eps_abs': eps_low,
                                          'eps_rel': eps_low,
                                          'warm_start': False,
                                          'polish': False,
                                          'max_iter': 40000,
                                          'eps_prim_inf': 1e-15,  # Disable infeas check
                                          'eps_dual_inf': 1e-15,
                                          'algebra': 'mkl',
                                          'solver_type': 'indirect',
                                          'alpha': 1.8,
                                          'sigma': 1e-2,
                                          'cg_max_iter': 5
       },
       QPFPGA: {'eps_abs': eps_low,
                'eps_rel': eps_low,
                'admm_max_iter': 40000,
                'alpha': 1.8,
                'sigma': 1e-2,
                'pcg_max_iter': 5,
                'measure_energy': True
       },
    GUROBI: {'TimeLimit': time_limit,
             'FeasibilityTol': eps_low,
             'OptimalityTol': eps_low,
             },
    GUROBI_high: {'TimeLimit': time_limit,
                  'FeasibilityTol': eps_high,
                  'OptimalityTol': eps_high,
                  },
    MOSEK: {'MSK_DPAR_OPTIMIZER_MAX_TIME': time_limit,
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': eps_low,   # Primal feasibility tolerance
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': eps_low,   # Dual feasibility tolerance
           },
    MOSEK_high: {'MSK_DPAR_OPTIMIZER_MAX_TIME': time_limit,
                 'MSK_DPAR_INTPNT_CO_TOL_PFEAS': eps_high,   # Primal feasibility tolerance
                 'MSK_DPAR_INTPNT_CO_TOL_DFEAS': eps_high,   # Dual feasibility tolerance
                },
    ECOS: {'abstol': eps_low,
           'reltol': eps_low},
    ECOS_high: {'abstol': eps_high,
                'reltol': eps_high},
    qpOASES: {}
}

for key in settings:
    settings[key]['verbose'] = False
    settings[key]['time_limit'] = time_limit