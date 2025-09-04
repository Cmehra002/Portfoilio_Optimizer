import cvxpy as cp
import numpy as np

class PortfolioOptimizer:
    """
    Defines and solves the portfolio optimization problem using cvxpy.
    """

    def __init__(self, mu: np.ndarray, cov: np.ndarray, w_benchmark: np.ndarray, w_old: np.ndarray):
        """
        Initializes the optimizer with necessary statistical data.
        """
        self.mu = mu
        self.cov = cov
        self.w_benchmark = w_benchmark
        self.w_old = w_old
        self.n = len(mu)
        self.problem = None
        self.w_var = None  # Save variable for extraction

    def define_problem(self, objective_str: str, constraints_str: list):
        """
        Dynamically defines the cvxpy problem from user-provided strings.
        """
        # Define the optimization variable
        w = cp.Variable(self.n, name="weights")
        # Safe context for eval
        context = {
            'cp': cp,
            'w': w,
            'mu': self.mu,
            'cov': self.cov,
            'w_benchmark': self.w_benchmark,
            'w_old': self.w_old
        }

        try:
            # Evaluate the objective and constraints strings
            objective = eval(objective_str, {"__builtins__": None}, context)
            constraints = [eval(c, {"__builtins__": None}, context) for c in constraints_str]
            self.problem = cp.Problem(objective, constraints)
            self.w_var = w
            print("Successfully defined the optimization problem.")
        except Exception as e:
            print(f"Error defining cvxpy problem from input strings: {e}")
            self.problem = None

    def solve(self, solver: str = 'SCS') -> tuple:
        """
        Solves the defined optimization problem.
        """
        if self.problem is None or self.w_var is None:
            return "Problem not defined", None
        print(f"Solving optimization problem using {solver} solver...")
        self.problem.solve(solver=solver)
        if self.problem.status in ["optimal", "optimal_inaccurate"]:
            print(f"Solver status: {self.problem.status}")
            return self.problem.status, self.w_var.value
        else:
            print(f"Optimization failed with status: {self.problem.status}")
            return self.problem.status, None
