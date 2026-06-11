import numpy as np
import scipy.sparse as spa
import cvxpy

from utils.general import random_sparse_nnz_per_col


class RandomQPExample(object):
    '''
    Random QP example
    '''
    def __init__(self, n, seed=1,nnz_per_col=None, build_cvxpy=True):
        '''
        Generate problem in QP format and CVXPY format
        '''
        # Set random seed
        np.random.seed(seed)

        # m = int(n * 10)
        m = n

        if nnz_per_col is None:
            nnz_per_col = 5

        # Generate problem data
        self.n = int(n)
        self.m = m

        actual_nnz = min(nnz_per_col, self.n)

        P = random_sparse_nnz_per_col(n, n, actual_nnz,
                          data_rvs=np.random.randn,
                          format='csc')
        self.P = P.dot(P.T).tocsc() + 1e-02 * spa.eye(n)
        self.q = np.random.randn(n)
        self.A = random_sparse_nnz_per_col(m, n, actual_nnz,
                           data_rvs=np.random.randn,
                           format='csc')
        v = np.random.randn(n)   # Fictitious solution
        delta = np.random.rand(m)  # To get inequality
        self.u = self.A@v + delta
        self.l = - np.inf * np.ones(m)  # self.u - np.random.rand(m)

        self.qp_problem = self._generate_qp_problem()
        if build_cvxpy:
            self.cvxpy_problem = self._generate_cvxpy_problem()
        else:
            self.cvxpy_problem = None

    @staticmethod
    def name():
        return 'Random QP'

    def _generate_qp_problem(self):
        '''
        Generate QP problem
        '''
        problem = {}
        problem['P'] = self.P
        problem['q'] = self.q
        problem['A'] = self.A
        problem['l'] = self.l
        problem['u'] = self.u
        problem['m'] = self.A.shape[0]
        problem['n'] = self.A.shape[1]

        return problem

    def _generate_cvxpy_problem(self):
        '''
        Generate QP problem
        '''
        x_var = cvxpy.Variable(self.n)
        objective = .5 * cvxpy.quad_form(x_var, self.P) + self.q * x_var
        constraints = [self.A * x_var <= self.u, self.A * x_var >= self.l]
        problem = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

        return problem

    def revert_cvxpy_solution(self):
        '''
        Get QP primal and duar variables from cvxpy solution
        '''

        variables = self.cvxpy_problem.variables()
        constraints = self.cvxpy_problem.constraints

        # primal solution
        x = variables[0].value

        # dual solution
        y = constraints[0].dual_value - constraints[1].dual_value

        return x, y
