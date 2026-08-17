"""
Tests for QUBO neuron selection: greedy returns exactly K, QUBO returns a valid K-subset.
"""
import numpy as np
import pytest

from qunart.qubo import QUBOSolver


class TestGreedySelection:
    def test_returns_exactly_k(self):
        rng = np.random.RandomState(42)
        importances = rng.rand(100)
        for k in [1, 10, 50, 100]:
            solver = QUBOSolver(method="greedy")
            selected = solver.solve(importances, k)
            assert len(selected) == k, f"Expected {k}, got {len(selected)}"

    def test_selects_top_k(self):
        importances = np.arange(20, dtype=float)
        solver = QUBOSolver(method="greedy")
        selected = solver.solve(importances, 5)
        expected = set(range(15, 20))
        assert set(selected) == expected

    def test_valid_indices(self):
        rng = np.random.RandomState(0)
        importances = rng.rand(50)
        solver = QUBOSolver(method="greedy")
        selected = solver.solve(importances, 20)
        assert all(0 <= idx < 50 for idx in selected)
        assert len(set(selected)) == 20  # all unique


class TestQUBOSelection:
    def test_returns_valid_k_subset(self):
        """QUBO returns exactly K unique valid indices."""
        rng = np.random.RandomState(42)
        n = 30
        importances = rng.rand(n)
        gate_weights = rng.rand(n, 16)
        up_weights = rng.rand(n, 16)

        solver = QUBOSolver(method="qubo", lambda_redundancy=0.1)
        selected = solver.solve(importances, 10,
                                gate_weights=gate_weights,
                                up_weights=up_weights)

        assert len(selected) == 10, f"Expected 10, got {len(selected)}"
        assert len(set(selected)) == 10, "Not all unique"
        assert all(0 <= idx < n for idx in selected), "Invalid indices"

    def test_qubo_without_weights_falls_to_cardinality_only(self):
        """Without gate/up weights, QUBO has no redundancy term but still works."""
        rng = np.random.RandomState(42)
        importances = rng.rand(20)

        solver = QUBOSolver(method="qubo", lambda_redundancy=0.1)
        selected = solver.solve(importances, 8)

        assert len(selected) == 8
        assert len(set(selected)) == 8
        assert all(0 <= idx < 20 for idx in selected)

    def test_qubo_large_n_with_prefilter(self):
        """For n > MAX_QUBO_SIZE, prefilter is used and result is still valid."""
        rng = np.random.RandomState(42)
        n = 600
        importances = rng.rand(n)
        gate_weights = rng.rand(n, 16)
        up_weights = rng.rand(n, 16)

        solver = QUBOSolver(method="qubo", lambda_redundancy=0.1)
        selected = solver.solve(importances, 100,
                                gate_weights=gate_weights,
                                up_weights=up_weights)

        assert len(selected) == 100
        assert len(set(selected)) == 100
        assert all(0 <= idx < n for idx in selected)

    def test_similarity_matrix_shape(self):
        """Sanity check: similarity matrix is correct shape."""
        rng = np.random.RandomState(42)
        gate = rng.rand(10, 8)
        up = rng.rand(10, 8)
        S = QUBOSolver._compute_similarity(gate, up)
        assert S.shape == (10, 10)
        assert np.allclose(np.diag(S), 0)
        assert (S >= 0).all()
