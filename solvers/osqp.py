import osqp
import time
from . import statuses as s
from .results import Results
from utils.general import is_qp_solution_optimal


class OSQPSolver(object):

    m = osqp.OSQP()
    STATUS_MAP = {osqp.constant('OSQP_SOLVED'): s.OPTIMAL,
                  osqp.constant('OSQP_MAX_ITER_REACHED'): s.MAX_ITER_REACHED,
                  osqp.constant('OSQP_PRIMAL_INFEASIBLE'): s.PRIMAL_INFEASIBLE,
                  osqp.constant('OSQP_DUAL_INFEASIBLE'): s.DUAL_INFEASIBLE}

    def __init__(self, settings={}):
        '''
        Initialize solver object by setting require settings
        '''
        self._settings = settings

    @property
    def settings(self):
        """Solver settings"""
        return self._settings

    def solve(self, example):
        '''
        Solve problem

        Args:
            problem: problem structure with QP matrices

        Returns:
            Results structure
        '''
        problem = example.qp_problem
        settings = self._settings.copy()
        high_accuracy = settings.pop('high_accuracy', None)
        algebra = settings.pop('algebra', None)
        cg_max_iter = settings.pop('cg_max_iter', None)
        debug = settings.pop('debug', False)

        if cg_max_iter is not None:
            settings['cg_max_iter'] = cg_max_iter

        # osqp-mkl registers the "mkl" algebra at import time.
        if algebra == 'mkl':
            try:
                import osqp_mkl  # noqa: F401
            except Exception as exc:
                raise RuntimeError(
                    'Failed to load osqp_mkl runtime. Ensure libmkl_rt.so.2 is discoverable via LD_LIBRARY_PATH.'
                ) from exc

        # Setup OSQP
        m = osqp.OSQP(algebra=algebra) if algebra is not None else osqp.OSQP()
        if debug:
            print(f'   [debug] OSQP setup start: algebra={algebra}, solver_type={settings.get("solver_type")}, eps_abs={settings.get("eps_abs")}, eps_rel={settings.get("eps_rel")}, cg_max_iter={settings.get("cg_max_iter")}, alpha={settings.get("alpha")}, sigma={settings.get("sigma")}')
        setup_start = time.perf_counter()
        m.setup(problem['P'], problem['q'], problem['A'], problem['l'],
                problem['u'],
                **settings)
        if debug:
            print(f'   [debug] OSQP setup done in {time.perf_counter() - setup_start:.3f}s')

        # Solve
        if debug:
            print('   [debug] OSQP solve start')
        solve_start = time.perf_counter()
        results = m.solve()
        if debug:
            print(f'   [debug] OSQP solve done in {time.perf_counter() - solve_start:.3f}s')
        status = self.STATUS_MAP.get(results.info.status_val, s.SOLVER_ERROR)

        if status in s.SOLUTION_PRESENT:
            if not is_qp_solution_optimal(problem,
                                          results.x,
                                          results.y,
                                          high_accuracy=high_accuracy):
                status = s.SOLVER_ERROR

        # Verify solver time
        if settings.get('time_limit') is not None:
            if results.info.run_time > settings.get('time_limit'):
                status = s.TIME_LIMIT

        return_results = Results(status,
                                 results.info.obj_val,
                                 results.x,
                                 results.y,
                                 results.info.run_time,
                                 results.info.iter)

        return_results.status_polish = results.info.status_polish
        return_results.setup_time = results.info.setup_time
        return_results.solve_time = results.info.solve_time
        return_results.update_time = results.info.update_time
        return_results.rho_updates = results.info.rho_updates

        return return_results
