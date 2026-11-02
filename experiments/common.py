
"""Common utilities for numerical linear algebra experiments."""

import numpy as np

from numerical_linear_alg import (
    gaussian_solve,
    lu_factor,
    lu_solve,
    residual,
    relative_residual,
    backward_error,
    condition_number,
)


def solve_with_lu(A, b):
    """Solve Ax = b using LU factorization."""

    P, L, U = lu_factor(A)
    return lu_solve(P, L, U, b)


def calculate_reference_metrics(A, x, b, x_exact):
    """Calculate numerical accuracy metrics independently."""

    r = b - A @ x
    residual_norm = np.linalg.norm(r)

    return {
        "solution_error": np.linalg.norm(x - x_exact),
        "residual_norm": residual_norm,
        "relative_residual": residual_norm / np.linalg.norm(b),
        "backward_error": residual_norm / (
            np.linalg.norm(A, 2) * np.linalg.norm(x, 2)
            + np.linalg.norm(b, 2)
        ),
    }


def validate_diagnostics(A, x, b, reference_metrics):
    """Validate package diagnostics against independent calculations."""

    np.testing.assert_allclose(
        residual(A, x, b),
        b - A @ x,
        rtol=1e-12,
        atol=1e-14,
    )

    np.testing.assert_allclose(
        relative_residual(A, x, b),
        reference_metrics["relative_residual"],
        rtol=1e-12,
        atol=1e-14,
    )

    np.testing.assert_allclose(
        backward_error(A, x, b),
        reference_metrics["backward_error"],
        rtol=1e-12,
        atol=1e-14,
    )


def run_solver_comparison(A, b, x_exact, title, additional_solvers=None):
    """Compare package solvers and optional experiment-specific solvers."""

    kappa_package = condition_number(A)
    kappa_numpy = np.linalg.cond(A)

    np.testing.assert_allclose(
        kappa_package,
        kappa_numpy,
        rtol=1e-12,
        atol=1e-14,
    )

    x_numpy = np.linalg.solve(A, b)

    solvers = {
        "Gaussian elimination (partial pivoting)": gaussian_solve,
        "LU factorization (partial pivoting)": solve_with_lu,
        "NumPy": np.linalg.solve,
    }

    if additional_solvers:
        solvers.update(additional_solvers)

    results = {}

    print(f"\n{title}")
    print("=" * 65)
    print("\nMatrix A:\n", A)
    print("\nExact solution:", x_exact)
    print("Right-hand side b:", b)
    print(f"\nCondition number: {kappa_package:.6e}")

    for name, solver in solvers.items():

        x = solver(A, b)

        metrics = calculate_reference_metrics(A, x, b, x_exact)

        validate_diagnostics(A, x, b, metrics)

        results[name] = {
            **metrics,
            "distance_from_numpy": np.linalg.norm(x - x_numpy),
        }

        print(f"\n{name}")
        print("-" * 50)
        print("Computed solution:", x)

        for metric, value in results[name].items():
            print(f"{metric}: {value:.6e}")

        print("Diagnostic validation: PASSED")

    return results, kappa_package
