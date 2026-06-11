import os
import time
import numpy as np
import scipy.sparse as spa
from multiprocessing import Pool, cpu_count
from itertools import repeat
import pandas as pd

from solvers.solvers import SOLVER_MAP
from utils.general import make_sure_path_exists

class RAPLMeter:
    def __init__(self):
        # intel-rapl:0 represents the entire CPU Package
        self.energy_file = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
        
    def read(self):
        try:
            with open(self.energy_file, 'r') as f:
                # Convert microjoules to Joules
                return float(f.read().strip()) / 1_000_000.0
        except:
            return np.nan
        

class LoadedExample(object):
    '''
    Mock class to simulate an Example instance by loading offline data from an .npz file.
    '''
    def __init__(self, problem_name, npz_file):
        self._name = problem_name
        data = np.load(npz_file)
        
        # Reconstruct CSC sparse matrices from raw arrays
        P = spa.csc_matrix((data['P_data'], data['P_indices'], data['P_indptr']), shape=data['P_shape'])
        A = spa.csc_matrix((data['A_data'], data['A_indices'], data['A_indptr']), shape=data['A_shape'])
        
        self.qp_problem = {
            'P': P,
            'A': A,
            'q': data['q'],
            'l': data['l'],
            'u': data['u'],
            'n': P.shape[1],
            'm': A.shape[0]
        }
        
    def name(self):
        return self._name


class Example(object):
    '''
    Offline Dataset Examples runner
    '''
    def __init__(self,
                name,
                dims,
                solvers,
                settings,
                output_folder,
                instance_map=None,
                 debug=False,
                 nnz_per_col=5,
                 build_cvxpy=False,
                 dataset_dir='dataset',
                 n_repeats=1):
        self.name = name
        self.dims = dims
        self.instance_map = instance_map or {}
        self.solvers = solvers
        self.settings = settings
        self.output_folder = output_folder
        self.debug = debug
        self.density = nnz_per_col
        self.build_cvxpy = build_cvxpy
        self.dataset_dir = dataset_dir
        self.n_repeats = n_repeats

    def solve(self, parallel=True):
        '''
        Solve problems using pre-loaded offline datasets.
        '''
        print("Solving %s (Density %s)" % (self.name, self.density))
        print("-----------------")

        if parallel:
            max_instances = max(
                (len(v) for v in self.instance_map.values()),
                default=1
            )
            pool = Pool(processes=min(max_instances, cpu_count()))

        # Iterate over all solvers
        for solver in self.solvers:
            settings = self.settings[solver]
            results_solver = []

            # Solution directory
            path = os.path.join('.', 'results', self.output_folder,
                                solver,
                                self.name)

            make_sure_path_exists(path)

            # Ensure full solver files are unique per density
            solver_file_name = os.path.join(path, f'full_density_{self.density}.csv')

            for n in self.dims:

                instances_list = self.instance_map.get(n, [])
                if not instances_list:
                    continue

                # Ensure individual files are unique per density
                n_file_name = os.path.join(path, f'n{n}_density_{self.density}.csv')

                if not os.path.isfile(n_file_name):
                    
                    if parallel and solver not in ['ECOS', 'ECOS_high', 'qpOASES']:
                        n_results = pool.starmap(self.solve_single_example,
                                                 zip(repeat(n),
                                                     instances_list,
                                                     repeat(solver),
                                                     repeat(settings)))
                    else:
                        n_results = []
                        for instance in instances_list:
                            n_results.append(
                                self.solve_single_example(n,
                                                          instance,
                                                          solver,
                                                          settings)
                                )

                    # Combine and Store
                    df = pd.concat(n_results)
                    df.to_csv(n_file_name, index=False)
                else:
                    df = pd.read_csv(n_file_name)

                results_solver.append(df)

            if not results_solver:
                continue

            df_solver = pd.concat(results_solver)
            df_solver.to_csv(solver_file_name, index=False)

        if parallel:
            pool.close()
            pool.join()

    def solve_single_example(self, dimension, instance_number, solver, base_settings):
        '''
        Loads an example from an .npz file, sweeps over adaptive_rho [0, 1], 
        and averages hardware profiling over n_repeats.
        '''
        # Resolve dataset file path
        prob_dir_name = self.name.lower().replace(' ', '_')
        npz_file = os.path.join(self.dataset_dir, prob_dir_name, f"density_{self.density}", f"n{dimension}_inst{instance_number}.npz")
        
        if self.debug:
            print(f'   [debug] Loading {self.name} instance from {npz_file}')

        if not os.path.exists(npz_file):
            raise FileNotFoundError(f"CRITICAL: Dataset file missing ({npz_file}).")

        example_instance = LoadedExample(self.name, npz_file)
        
        P = example_instance.qp_problem['P']
        A = example_instance.qp_problem['A']
        
        # --- PROBLEM DATA ---
        n_size = example_instance.qp_problem['n']
        m_size = example_instance.qp_problem['m']
        nnz_P = P.nnz
        nnz_A = A.nnz

        # We will collect multiple rows for this single matrix (rho=0 and rho=1)
        csv_rows = []

        # --- SWEEP ADAPTIVE RHO ---
        for rho_val in [0, 1]:
            # Copy settings so we don't permanently overwrite the global dictionary
            settings = base_settings.copy()
            settings['adaptive_rho'] = rho_val
            
            # --- SOLVER CONFIGURATION ---
            eps_abs = settings.get('eps_abs', np.nan)
            eps_rel = settings.get('eps_rel', np.nan)
            max_iter = settings.get('max_iter', np.nan)
            if np.isnan(max_iter):
                max_iter = settings.get('admm_max_iter', np.nan)
            cg_max_iter = settings.get('cg_max_iter', np.nan)
            if np.isnan(cg_max_iter):
                cg_max_iter = settings.get('pcg_max_iter', np.nan)
            alpha = settings.get('alpha', np.nan)
            sigma = settings.get('sigma', np.nan)

            print(f" - Solving {self.name} (n={dimension}, inst={instance_number}, density={self.density}, rho={rho_val}) with {solver}")

            try:
                best_results = None
                best_solver_obj = None
                
                total_run_time = 0.0
                total_setup_time = 0.0
                total_solve_time = 0.0
                total_update_time = 0.0
                
                total_energy = 0.0
                valid_energy_reads = 0
                meter = RAPLMeter()
                
                # --- HARDWARE PROFILING LOOP ---
                for run_idx in range(getattr(self, 'n_repeats', 1)):
                    s = SOLVER_MAP[solver](settings)
                    
                    # 1. Measure Energy BEFORE
                    start_energy = meter.read()
                    
                    # 2. SOLVE
                    results = s.solve(example_instance)
                    
                    # 3. Measure Energy AFTER
                    end_energy = meter.read()
                    
                    # Accumulate Timings
                    total_run_time += results.run_time
                    if solver[:4] == 'OSQP':
                        total_setup_time += results.setup_time
                        total_solve_time += results.solve_time
                        total_update_time += getattr(results, 'update_time', 0.0)
                    
                    # Accumulate Energy
                    if not np.isnan(start_energy) and not np.isnan(end_energy):
                        diff = end_energy - start_energy
                        # Guard against RAPL register overflow/reset
                        if diff >= 0:
                            total_energy += diff
                            valid_energy_reads += 1
                            
                    best_results = results
                    best_solver_obj = s
                
                # --- CALCULATE AVERAGES ---
                status = best_results.status
                run_time = total_run_time / self.n_repeats
                niter = best_results.niter
                obj_val = best_results.obj_val
                
                # Energy and Power Averages
                if valid_energy_reads > 0 and run_time > 0:
                    energy_per_solve = total_energy / valid_energy_reads
                    avg_power = energy_per_solve / run_time
                else:
                    energy_per_solve = np.nan
                    avg_power = np.nan
                                
                # Steal residuals and CG iterations from the raw OSQP/QPFPGA object
                raw_info = getattr(best_solver_obj, 'results', None)
                if raw_info and hasattr(raw_info, 'info'):
                    pri_res = getattr(raw_info.info, 'pri_res', np.nan)
                    dua_res = getattr(raw_info.info, 'dua_res', np.nan)
                    cg_iters = getattr(raw_info.info, 'cg_iters', np.nan)
                elif raw_info and hasattr(raw_info, 'primal_residual'): # QPFPGA branch
                    pri_res = getattr(raw_info, 'primal_residual', np.nan)
                    dua_res = getattr(raw_info, 'dual_residual', np.nan)
                    cg_iters = raw_info.extra_stats.get('pcg_iters', np.nan)
                else:
                    pri_res, dua_res, cg_iters = np.nan, np.nan, np.nan

                if np.isnan(pri_res) or np.isnan(dua_res):
                    x_val = best_results.x
                    y_val = best_results.y
                    
                    # Ensure we have valid vectors (not None) before doing math
                    if x_val is not None and y_val is not None and len(x_val) > 0:
                        l_bound = example_instance.qp_problem['l']
                        u_bound = example_instance.qp_problem['u']
                        q_val = example_instance.qp_problem['q']
                        
                        # Primal Residual: || Ax - clip(Ax, l, u) ||_inf
                        Ax = A.dot(x_val)
                        z_val = np.clip(Ax, l_bound, u_bound)
                        pri_res = np.max(np.abs(Ax - z_val))
                        
                        # Dual Residual: || Px + q + A^T y ||_inf
                        Px = P.dot(x_val)
                        ATy = A.T.dot(y_val)
                        dua_res = np.max(np.abs(Px + q_val + ATy))
                
                # OSQP Timings
                setup_time = total_setup_time / self.n_repeats if solver[:4] == 'OSQP' else np.nan
                solve_time = total_solve_time / self.n_repeats if solver[:4] == 'OSQP' else np.nan
                update_time = total_update_time / self.n_repeats if solver[:4] == 'OSQP' else np.nan
                status_polish = best_results.status_polish if solver[:4] == 'OSQP' else np.nan
                rho_updates = best_results.rho_updates if solver[:4] == 'OSQP' else np.nan
                
                # FPGA metrics extraction
                fpga_cpp_total = np.nan
                fpga_data_prep = np.nan
                fpga_hw_exec = np.nan
                fpga_energy_core = np.nan
                fpga_energy_aux = np.nan
                fpga_energy_total = np.nan
                fpga_energy_board = np.nan
                
                if solver == 'QPFPGA' and raw_info is not None:
                    fpga_cpp_total = raw_info.extra_stats.get('total_cpp_time_s', np.nan)
                    fpga_data_prep = raw_info.extra_stats.get('setup_time_ms', np.nan) / 1000.0 if not np.isnan(raw_info.extra_stats.get('setup_time_ms', np.nan)) else np.nan
                    fpga_hw_exec = raw_info.solve_time_s
                    fpga_energy_core = raw_info.extra_stats.get('core_energy_j', np.nan)
                    fpga_energy_aux = raw_info.extra_stats.get('aux_energy_j', np.nan)
                    fpga_energy_total = raw_info.extra_stats.get('fpga_energy_j', np.nan)
                    fpga_energy_board = raw_info.extra_stats.get('board_energy_j', np.nan)

            except Exception as e:
                print(f"   [!] Solver {solver} internally failed/crashed: {e}")
                status = 'Solver Error'
                run_time, niter, obj_val = np.nan, np.nan, np.nan
                pri_res, dua_res, cg_iters = np.nan, np.nan, np.nan
                setup_time, solve_time, update_time = np.nan, np.nan, np.nan
                status_polish, rho_updates = np.nan, np.nan
                energy_per_solve, avg_power = np.nan, np.nan
                fpga_cpp_total, fpga_data_prep, fpga_hw_exec = np.nan, np.nan, np.nan
                fpga_energy_core, fpga_energy_aux, fpga_energy_total, fpga_energy_board = np.nan, np.nan, np.nan, np.nan

            # --- BUILD THE UNIFIED CSV DICTIONARY FOR THIS RHO VALUE ---
            row_dict = {
                'class': self.name,
                'solver': solver,
                'density': self.density,
                'status': status,
                'n': n_size,
                'm': m_size,
                'nnz_A': nnz_A,
                'nnz_P': nnz_P,
                'eps_abs': eps_abs,
                'eps_rel': eps_rel,
                'adaptive_rho': rho_val,
                'max_iter': max_iter,
                'cg_max_iter': cg_max_iter,
                'alpha': alpha,
                'sigma': sigma,
                'run_time': run_time,
                'setup_time': setup_time,
                'solve_time': solve_time,
                'update_time': update_time,
                'fpga_cpp_total': fpga_cpp_total,
                'fpga_data_prep': fpga_data_prep,
                'fpga_hw_exec': fpga_hw_exec,
                'iter_admm': niter,
                'iter_cg': cg_iters,
                'rho_updates': rho_updates,
                'status_polish': status_polish,
                'obj_val': obj_val,
                'pri_res': pri_res,
                'dua_res': dua_res,
                'energy_solve_cpu': energy_per_solve,
                'avg_power_cpu': avg_power,
                'fpga_energy_core': fpga_energy_core,
                'fpga_energy_aux': fpga_energy_aux,
                'fpga_energy_total': fpga_energy_total,
                'fpga_energy_board': fpga_energy_board
            }
            csv_rows.append(row_dict)

        # Return a DataFrame containing both rho=0 and rho=1 tests
        return pd.DataFrame(csv_rows)