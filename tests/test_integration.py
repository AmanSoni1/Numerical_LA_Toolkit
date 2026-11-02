"""Integration tests for experiments and numerical linear algebra utilities."""

import numpy as np
import pytest

from experiments.common import (
    calculate_reference_metrics,
    run_solver_comparison,
    solve_with_lu,
    validate_diagnostics,
)
from experiments.gaussian_no_pivoting import gaussian_solve_without_pivoting
from experiments.hilbert import create_hilbert_system
from experiments.pivoting import create_system as create_pivoting_system
from experiments.well_conditioned import create_system as create_well_conditioned_system


# ---------------------------------------------------------
# 1. Gaussian elimination without partial pivoting
# ---------------------------------------------------------

def test_gaussian_without_pivoting_solves_system():
    A = np.array([
        [4.0, 1.0, 1.0],
        [1.0, 3.0, 0.0],
        [1.0, 0.0, 2.0],
    ])

    x_exact = np.array([1.0, 2.0, 3.0])
    b = A @ x_exact

    x = gaussian_solve_without_pivoting(A, b)

    assert np.allclose(x, x_exact)


def test_gaussian_without_pivoting_agrees_with_numpy():
    A = np.array([
        [3.0, 1.0],
        [1.0, 2.0],
    ])

    b = np.array([7.0, 5.0])

    x = gaussian_solve_without_pivoting(A, b)

    assert np.allclose(x, np.linalg.solve(A, b))


def test_gaussian_without_pivoting_detects_zero_pivot():
    A = np.array([
        [0.0, 1.0],
        [1.0, 1.0],
    ])

    b = np.array([1.0, 2.0])

    with pytest.raises(np.linalg.LinAlgError):
        gaussian_solve_without_pivoting(A, b)


def test_gaussian_without_pivoting_does_not_modify_inputs():
    A = np.array([
        [3.0, 1.0],
        [1.0, 2.0],
    ])

    b = np.array([7.0, 5.0])

    A_original = A.copy()
    b_original = b.copy()

    gaussian_solve_without_pivoting(A, b)

    assert np.array_equal(A, A_original)
    assert np.array_equal(b, b_original)


# ---------------------------------------------------------
# 2. Experiment system construction
# ---------------------------------------------------------

def test_well_conditioned_system_has_known_solution():
    A, b, x_exact = create_well_conditioned_system()

    assert A.shape == (3, 3)
    assert np.allclose(b, A @ x_exact)
    assert np.allclose(
        np.linalg.solve(A, b),
        x_exact,
    )


def test_pivoting_system_has_expected_structure():
    A, b, x_exact = create_pivoting_system()

    assert A.shape == (2, 2)
    assert A[0, 0] == pytest.approx(1e-12)
    assert np.allclose(b, A @ x_exact)
    assert np.allclose(x_exact, np.ones(2))


def test_pivoting_experiment_demonstrates_multiplier_difference():
    A, _, _ = create_pivoting_system()

    multiplier_without_pivoting = A[1, 0] / A[0, 0]
    multiplier_with_pivoting = A[0, 0] / A[1, 0]

    assert multiplier_without_pivoting == pytest.approx(1e12)
    assert multiplier_with_pivoting == pytest.approx(1e-12)


# ---------------------------------------------------------
# 3. Hilbert matrix construction
# ---------------------------------------------------------

@pytest.mark.parametrize("n", [1, 3, 5, 8])
def test_hilbert_system_construction(n):
    H, b, x_exact = create_hilbert_system(n)

    expected_H = np.fromfunction(
        lambda i, j: 1.0 / (i + j + 1),
        (n, n),
        dtype=float,
    )

    assert H.shape == (n, n)
    assert np.allclose(H, expected_H)
    assert np.allclose(x_exact, np.ones(n))
    assert np.allclose(b, H @ x_exact)


# ---------------------------------------------------------
# 4. Common numerical utilities
# ---------------------------------------------------------

def test_solve_with_lu():
    A, b, x_exact = create_well_conditioned_system()

    x = solve_with_lu(A, b)

    assert np.allclose(x, x_exact)


def test_calculate_reference_metrics():
    A, b, x_exact = create_well_conditioned_system()

    x = x_exact + np.array([0.01, -0.02, 0.01])

    metrics = calculate_reference_metrics(A, x, b, x_exact)

    residual_norm = np.linalg.norm(b - A @ x)

    expected_solution_error = np.linalg.norm(x - x_exact)
    expected_relative_residual = residual_norm / np.linalg.norm(b)

    expected_backward_error = residual_norm / (
        np.linalg.norm(A, 2) * np.linalg.norm(x, 2)
        + np.linalg.norm(b, 2)
    )

    assert metrics["solution_error"] == pytest.approx(
        expected_solution_error
    )

    assert metrics["residual_norm"] == pytest.approx(
        residual_norm
    )

    assert metrics["relative_residual"] == pytest.approx(
        expected_relative_residual
    )

    assert metrics["backward_error"] == pytest.approx(
        expected_backward_error
    )


def test_validate_diagnostics():
    A, b, x_exact = create_well_conditioned_system()

    metrics = calculate_reference_metrics(A, x_exact, b, x_exact)

    # Should complete without an assertion error.
    validate_diagnostics(A, x_exact, b, metrics)


# ---------------------------------------------------------
# 5. Solver comparison integration
# ---------------------------------------------------------

def test_run_solver_comparison():
    A, b, x_exact = create_well_conditioned_system()

    results, condition_number = run_solver_comparison(
        A,
        b,
        x_exact,
        "TEST: WELL-CONDITIONED SYSTEM",
    )

    expected_solvers = {
        "Gaussian elimination (partial pivoting)",
        "LU factorization (partial pivoting)",
        "NumPy",
    }

    assert set(results.keys()) == expected_solvers
    assert condition_number >= 1.0

    for metrics in results.values():
        assert metrics["solution_error"] < 1e-12
        assert metrics["residual_norm"] < 1e-12
        assert metrics["relative_residual"] < 1e-12
        assert metrics["backward_error"] < 1e-12