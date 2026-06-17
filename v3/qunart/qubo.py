import numpy as np
from typing import Optional


class QUBOSolver:
    """
    Binary selection: minimize  sum_i I_i * (1 - x_i) + lambda * (sum_i x_i - K)^2.

    Falls back to greedy if `dwave-neal` is not installed.
    """

    def __init__(self, method: str = "greedy", penalty: float = 1.0):
        self.method = method.lower()
        self.penalty = penalty

    def solve(self, importances: np.ndarray, target_keep: int) -> np.ndarray:
        n = len(importances)
        target_keep = max(1, min(target_keep, n))

        if self.method == "greedy":
            return np.argsort(importances)[-target_keep:]

        if self.method == "qubo":
            return self._solve_qubo(importances, target_keep)

        raise ValueError(f"Unknown QUBO method: {self.method}")

    def _solve_qubo(self, importances: np.ndarray, target_keep: int) -> np.ndarray:
        try:
            import neal

            n = len(importances)
            Q = np.zeros((n, n))
            for i in range(n):
                Q[i, i] = -importances[i] + self.penalty * (1 - 2 * target_keep)
            for i in range(n):
                for j in range(i + 1, n):
                    Q[i, j] = 2 * self.penalty

            sampler = neal.SimulatedAnnealingSampler()
            sampleset = sampler.sample_qubo(Q, num_reads=200)
            best = sampleset.first.sample
            return np.array([i for i, v in best.items() if v == 1])
        except ImportError:
            return np.argsort(importances)[-target_keep:]

