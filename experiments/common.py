
"""Common utilities for numerical linear algebra experiments."""

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from numerical_linear_alg import (
    GaussianEliminationSolver,
    LinearSolver,
    LUSolver,
    MatrixDiagnostics,
    SolutionDiagnostics,
)


class NumpySolver(LinearSolver):
    """Reference solver: NumPy's LAPACK-based np.linalg.solve."""

    name = "NumPy"

    def solve(self, A, b):
        return np.linalg.solve(A, b)


class Experiment:
    """Base class for every numerical experiment."""

    title = "EXPERIMENT"

    def run(self):
        """Execute the experiment."""
        raise NotImplementedError("Subclasses must implement run().")

    def main(self):
        """Entry point used by python -m experiments.<name>."""
        return self.run()


class SolverComparison:
    """
    Compare the package solvers (and optional extra solvers) on one system
    A x = b with a known exact solution.
    """

    GAUSSIAN = GaussianEliminationSolver.name
    LU = LUSolver.name
    NUMPY = NumpySolver.name

    def __init__(self, A, b, x_exact, title, additional_solvers=None):
        self.A = A
        self.b = b
        self.x_exact = x_exact
        self.title = title

        self.solvers = {
            self.GAUSSIAN: GaussianEliminationSolver(),
            self.LU: LUSolver(),
            self.NUMPY: NumpySolver(),
        }

        if additional_solvers:
            self.solvers.update(additional_solvers)

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------
    def calculate_reference_metrics(self, A, x, b, x_exact):
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

    def validate_diagnostics(self, A, x, b, reference_metrics):
        """Validate package diagnostics against independent calculations."""

        diagnostics = SolutionDiagnostics(A, x, b)

        np.testing.assert_allclose(
            diagnostics.residual(),
            b - A @ x,
            rtol=1e-12,
            atol=1e-14,
        )

        np.testing.assert_allclose(
            diagnostics.relative_residual(),
            reference_metrics["relative_residual"],
            rtol=1e-12,
            atol=1e-14,
        )

        np.testing.assert_allclose(
            diagnostics.backward_error(),
            reference_metrics["backward_error"],
            rtol=1e-12,
            atol=1e-14,
        )

    # ------------------------------------------------------------------
    # Main routine
    # ------------------------------------------------------------------
    def run(self):
        """Run every solver; return (results, condition_number)."""

        A, b, x_exact = self.A, self.b, self.x_exact

        kappa_package = MatrixDiagnostics(A).condition_number()
        kappa_numpy = np.linalg.cond(A)

        np.testing.assert_allclose(
            kappa_package,
            kappa_numpy,
            rtol=1e-12,
            atol=1e-14,
        )

        x_numpy = np.linalg.solve(A, b)

        results = {}

        print(f"\n{self.title}")
        print("=" * 65)
        print("\nMatrix A:\n", A)
        print("\nExact solution:", x_exact)
        print("Right-hand side b:", b)
        print(f"\nCondition number: {kappa_package:.6e}")

        for name, solver in self.solvers.items():

            x = solver.solve(A, b)

            metrics = self.calculate_reference_metrics(
                A, x, b, x_exact
            )

            self.validate_diagnostics(A, x, b, metrics)

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


# ----------------------------------------------------------------------
# Backward-compatible functional API
# ----------------------------------------------------------------------
def solve_with_lu(A, b):
    """Solve Ax = b using LU factorization."""
    return LUSolver().solve(A, b)


def calculate_reference_metrics(A, x, b, x_exact):
    return SolverComparison(
        None, None, None, None
    ).calculate_reference_metrics(A, x, b, x_exact)


def validate_diagnostics(A, x, b, reference_metrics):
    SolverComparison(
        None, None, None, None
    ).validate_diagnostics(A, x, b, reference_metrics)


def run_solver_comparison(A, b, x_exact, title, additional_solvers=None):
    """Compare package solvers and optional experiment-specific solvers."""
    return SolverComparison(
        A, b, x_exact, title, additional_solvers
    ).run()
