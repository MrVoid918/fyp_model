import numpy as np


class Assignment:

    def __init__(self) -> None:
        pass

    def __call__(self, cost_matrix):
        try:
            import lap
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
            return np.array([[y[i],i] for i in x if i >= 0]) #
        except ImportError:
            from scipy.optimize import linear_sum_assignment
            x, y = linear_sum_assignment(cost_matrix)
            return np.array(list(zip(x, y)))