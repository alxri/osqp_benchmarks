import time
import numpy as np
import scipy.sparse as sp
from . import statuses as s
from .results import Results
import qpfpga
from qpfpga.backend import as_osqp_problem

# --- GLOBAL SINGLETON CACHE ---
# Ensures the FPGA shared library is loaded exactly ONCE per process
_GLOBAL_QPFPGA_BACKEND = None

class QPFPGA_Solver(object):

    STATUS_MAP = {
        "optimal": s.OPTIMAL,
        "optimal_inaccurate": s.OPTIMAL_INACCURATE,
        "user_limit": s.TIME_LIMIT,
        "infeasible": s.PRIMAL_INFEASIBLE,
        "infeasible_inaccurate": s.PRIMAL_INFEASIBLE,
        "unbounded": s.DUAL_INFEASIBLE,
        "unbounded_inaccurate": s.DUAL_INFEASIBLE,
        "solver_error": s.SOLVER_ERROR,
        "not_implemented": s.SOLVER_ERROR,
    }

    def __init__(self, settings={}):
        self._settings = settings

    @property
    def settings(self):
        return self._settings
        
    def _is_fpga_solution_optimal(self, problem, results, eps_abs, eps_rel):
        """
        Custom optimality check using the raw 32-bit residuals calculated 
        by the FPGA, scaled against OSQP's dynamic threshold math.
        """
        if results.x is None or results.y is None:
            return False

        n = problem['n']
        m = problem['m']
        
        # Calculate dynamic threshold scaling factors
        Ax = problem['A'].dot(results.x)
        z = np.clip(Ax, problem['l'], problem['u'])
        max_Ax_z = max(np.linalg.norm(Ax, np.inf), np.linalg.norm(z, np.inf))
        
        Px = problem['P'].dot(results.x)
        ATy = problem['A'].T.dot(results.y)
        q = problem['q']
        max_Px_ATy_q = max(np.linalg.norm(Px, np.inf), 
                           np.linalg.norm(ATy, np.inf), 
                           np.linalg.norm(q, np.inf))

        # Calculate the dynamic OSQP tolerances
        eps_primal = eps_abs * np.sqrt(m) + eps_rel * max_Ax_z
        eps_dual = eps_abs * np.sqrt(n) + eps_rel * max_Px_ATy_q

        pri_res = results.primal_residual
        dua_res = results.dual_residual

        primal_ok = pri_res <= eps_primal
        dual_ok = dua_res <= eps_dual
        
        is_optimal = primal_ok and dual_ok
        is_inaccurate = (pri_res <= 10 * eps_primal) and (dua_res <= 10 * eps_dual)
        
        return is_optimal or is_inaccurate

    def solve(self, example):
        global _GLOBAL_QPFPGA_BACKEND
        
        problem = example.qp_problem
        settings = self._settings.copy()
        
        debug = settings.pop('debug', False)
        
        eps_abs_val = float(settings.get("eps_abs", 1e-3))
        eps_rel_val = float(settings.get("eps_rel", 1e-3))
        
        import qpfpga.data as qdata
        options = qdata.QPSolverOptions(
            sigma=float(settings.get("sigma", 1e-2)),
            alpha=float(settings.get("alpha", 1.8)),
            eps_abs=eps_abs_val,
            eps_rel=eps_rel_val,
            pcg_tol_fraction=float(settings.get("pcg_tol_fraction", 1.0)),
            admm_max_iter=int(settings.get("admm_max_iter", 40000)),
            pcg_max_iter=int(settings.get("pcg_max_iter", 5)),
            adaptive_rho=bool(settings.get("adaptive_rho", False)),
            measure_energy=bool(settings.get("measure_energy", True)),
        )

        qp_data = qdata.QPData(
            P=problem['P'],
            q=problem['q'],
            A=problem['A'],
            l=problem['l'],
            u=problem['u']
        )
        
        if debug:
            print("   [debug] QPFPGA solve start")
            
        # --- LOAD CACHED BACKEND ---
        if _GLOBAL_QPFPGA_BACKEND is None:
            _GLOBAL_QPFPGA_BACKEND = qpfpga.backend.default_backend()
            
        solver = _GLOBAL_QPFPGA_BACKEND
        
        t0 = time.perf_counter()
        results = solver.solve(qp_data, options)
        run_time = time.perf_counter() - t0
        
        self.results = results
        
        if debug:
            print(f"   [debug] QPFPGA solve done in {run_time:.3f}s")
            
        # --- COMPUTE OBJECTIVE VALUE ON CPU ---
        final_obj_val = results.obj_val
        if (final_obj_val is None or final_obj_val == 0.0) and results.x is not None:
            x = results.x
            Px = problem['P'].dot(x)
            final_obj_val = float(0.5 * x.dot(Px) + problem['q'].dot(x))
            
        status = self.STATUS_MAP.get(results.status, s.SOLVER_ERROR)

        # --- CUSTOM FPGA OPTIMALITY CHECK ---
        if status in [s.OPTIMAL, s.OPTIMAL_INACCURATE]:
            is_valid = self._is_fpga_solution_optimal(problem, results, eps_abs_val, eps_rel_val)
            if not is_valid:
                if debug:
                    print("   [debug] FPGA returned optimal, but hardware residuals exceeded dynamic thresholds.")
                status = s.SOLVER_ERROR
                
        if settings.get('time_limit') is not None:
            if run_time > settings.get('time_limit'):
                status = s.TIME_LIMIT

        return_results = Results(status,
                                 final_obj_val,
                                 results.x,
                                 results.y,
                                 run_time,
                                 results.num_iters)
                                 
        return_results.setup_time = 0.0
        return_results.solve_time = run_time
        return_results.update_time = 0.0
        return_results.rho_updates = 0
        return_results.status_polish = 0

        return return_results