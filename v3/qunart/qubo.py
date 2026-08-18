import numpy as np
from typing import Optional


class QUBOSolver:
    """
    Binary neuron selection via QUBO formulation.

    Objective:
      minimise  -Σ I_i·x_i  +  λ_red · Σ_{i<j} S_ij·x_i·x_j  +  λ_card · (Σ x_i − K)²

    where S_ij is the cosine similarity between the concatenated
    [gate_i ; up_i] weight rows (clamped to ≥ 0), I_i is the importance
    of neuron i, and K is the target number of neurons to keep.

    The redundancy term penalises keeping two highly-correlated neurons,
    making the QUBO formulation genuinely quadratic (not just a cardinality
    constraint that reduces to top-K).

    Falls back to greedy (top-K by importance) if dwave-neal is not installed.
    """

    # Cap for tractability: if more candidates than this, pre-select the
    # top MAX_QUBO_SIZE by importance and solve QUBO on those only.
    MAX_QUBO_SIZE = 512

    def __init__(
        self,
        method: str = "greedy",
        penalty: float = 1.0,
        lambda_redundancy: float = 0.1,
    ):
        self.method = method.lower()
        self.penalty = penalty
        self.lambda_redundancy = lambda_redundancy

    def solve(
        self,
        importances: np.ndarray,
        target_keep: int,
        gate_weights: Optional[np.ndarray] = None,
        up_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Select target_keep neurons from n candidates.

        Args:
            importances: (n,) importance scores.
            target_keep: number of neurons to keep.
            gate_weights: (n, h) gate projection rows, for redundancy.
            up_weights: (n, h) up projection rows, for redundancy.

        Returns:
            (target_keep,) array of selected indices.
        """
        n = len(importances)
        target_keep = max(1, min(target_keep, n))

        if self.method == "greedy":
            return self._solve_greedy(importances, target_keep)

        if self.method == "qubo":
            return self._solve_qubo(
                importances, target_keep,
                gate_weights=gate_weights,
                up_weights=up_weights,
            )

        raise ValueError(f"Unknown QUBO method: {self.method}")

    def _solve_greedy(self, importances: np.ndarray, target_keep: int) -> np.ndarray:
        return np.argsort(importances)[-target_keep:]

    def _solve_qubo(
        self,
        importances: np.ndarray,
        target_keep: int,
        gate_weights: Optional[np.ndarray] = None,
        up_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n = len(importances)

        # If n is larger than the tractable QUBO capacity, solve on the contested band
        if n > self.MAX_QUBO_SIZE:
            return self._solve_qubo_with_prefilter(
                importances, target_keep,
                gate_weights=gate_weights,
                up_weights=up_weights,
            )

        return self._solve_qubo_direct(
            importances, target_keep,
            gate_weights=gate_weights,
            up_weights=up_weights,
        )

    def _solve_qubo_direct(
        self,
        importances: np.ndarray,
        target_keep: int,
        gate_weights: Optional[np.ndarray] = None,
        up_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve QUBO directly on a candidate set of size <= MAX_QUBO_SIZE."""
        n = len(importances)
        target_keep = max(1, min(target_keep, n))

        # Build the QUBO matrix
        Q = self._build_qubo_matrix(
            importances, target_keep,
            gate_weights=gate_weights,
            up_weights=up_weights,
        )

        try:
            import neal
            sampler = neal.SimulatedAnnealingSampler()
            sampleset = sampler.sample_qubo(
                {(i, j): Q[i, j] for i in range(n) for j in range(i, n) if Q[i, j] != 0},
                num_reads=200,
            )
            best = sampleset.first.sample
            selected = np.array([i for i, v in best.items() if v == 1])
        except ImportError:
            # Fallback to greedy if dwave-neal not installed
            selected = self._solve_greedy(importances, target_keep)

        # Ensure exactly target_keep are selected
        if len(selected) > target_keep:
            # Keep the highest importance ones
            sub_imp = importances[selected]
            keep_idx = np.argsort(sub_imp)[-target_keep:]
            selected = selected[keep_idx]
        elif len(selected) < target_keep:
            # Add highest-importance unselected neurons
            mask = np.ones(n, dtype=bool)
            mask[selected] = False
            remaining = np.where(mask)[0]
            extra = remaining[np.argsort(importances[remaining])[-(target_keep - len(selected)):]]
            selected = np.concatenate([selected, extra])

        return selected

    def _solve_qubo_with_prefilter(
        self,
        importances: np.ndarray,
        target_keep: int,
        gate_weights: Optional[np.ndarray] = None,
        up_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        For large candidate counts (n > MAX_QUBO_SIZE), isolate the contested
        boundary band around the cutoff threshold, take clear winners greedily,
        and solve QUBO on the contested candidate band.
        """
        n = len(importances)
        sorted_indices = np.argsort(importances)  # ascending order

        band_size = min(self.MAX_QUBO_SIZE, n)

        # Sure winners above contested boundary
        sure_keep_count = max(0, target_keep - band_size // 2)
        sure_keep_count = min(sure_keep_count, n - band_size)

        sure_keepers = (
            sorted_indices[n - sure_keep_count:]
            if sure_keep_count > 0
            else np.array([], dtype=int)
        )

        quota_needed = target_keep - sure_keep_count

        # Contested candidate window
        contested_start = max(0, n - sure_keep_count - band_size)
        contested_end = n - sure_keep_count
        contested_indices = sorted_indices[contested_start:contested_end]

        sub_imp = importances[contested_indices]
        sub_gate = gate_weights[contested_indices] if gate_weights is not None else None
        sub_up = up_weights[contested_indices] if up_weights is not None else None

        sub_selected = self._solve_qubo_direct(
            sub_imp, quota_needed,
            gate_weights=sub_gate,
            up_weights=sub_up,
        )

        qubo_selected = contested_indices[sub_selected]
        return np.concatenate([sure_keepers, qubo_selected])

    def _build_qubo_matrix(
        self,
        importances: np.ndarray,
        target_keep: int,
        gate_weights: Optional[np.ndarray] = None,
        up_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build the QUBO matrix Q such that:
          E(x) = Σ_i Q[i,i]·x_i + Σ_{i<j} Q[i,j]·x_i·x_j
        """
        n = len(importances)
        Q = np.zeros((n, n), dtype=np.float64)

        # Normalise importances to [0, 1] for stable penalty scaling
        imp = importances.astype(np.float64)
        imp_max = imp.max()
        if imp_max > 0:
            imp_norm = imp / imp_max
        else:
            imp_norm = imp

        # Diagonal: linear importance term + cardinality penalty diagonal
        #   -I_i + λ_card * (1 - 2K)
        lam_card = self.penalty
        for i in range(n):
            Q[i, i] = -imp_norm[i] + lam_card * (1 - 2 * target_keep)

        # Off-diagonal: cardinality penalty
        #   2 * λ_card  for all i < j
        for i in range(n):
            for j in range(i + 1, n):
                Q[i, j] = 2 * lam_card

        # Off-diagonal: redundancy penalty (the genuine pairwise interaction)
        if gate_weights is not None and up_weights is not None and self.lambda_redundancy > 0:
            S = self._compute_similarity(gate_weights, up_weights)
            for i in range(n):
                for j in range(i + 1, n):
                    Q[i, j] += self.lambda_redundancy * S[i, j]

        return Q

    @staticmethod
    def _compute_similarity(
        gate_weights: np.ndarray,
        up_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity between neurons using
        concatenated [gate_i ; up_i] weight vectors. Clamp negatives to 0.

        Returns: (n, n) symmetric matrix with 0 on diagonal.
        """
        # Concatenate gate and up rows: (n, 2*h)
        combined = np.concatenate([gate_weights, up_weights], axis=1).astype(np.float64)
        # Normalise rows
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normed = combined / norms
        # Cosine similarity matrix
        S = normed @ normed.T
        # Clamp negatives and zero diagonal
        np.maximum(S, 0, out=S)
        np.fill_diagonal(S, 0)
        return S
