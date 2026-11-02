
"""Experiment 2: Investigate the importance of partial pivoting.

Compare Gaussian elimination with and without partial pivoting
using a linear system with a very small initial pivot.
"""

import numpy as np

from experiments.common import run_solver_comparison
from experiments.gaussian_no_pivoting import gaussian_solve_without_pivoting


def create_system():
    """Create the linear system specified for the pivoting experiment."""

    A = np.array([
        [1e-12, 1.0],
        [1.0, 1.0],
    ])

    x_exact = np.array([1.0, 1.0])

    b = A @ x_exact

    return A, b, x_exact


def main():

    A, b, x_exact = create_system()

    # Calculate the elimination multipliers.
    multiplier_without_pivoting = A[1, 0] / A[0, 0]
    multiplier_with_pivoting = A[0, 0] / A[1, 0]

    print("EXPERIMENT 2: IMPORTANCE OF PARTIAL PIVOTING")
    print("=" * 60)

    print("\nInitial pivot:", A[0, 0])

    print(
        "Multiplier without pivoting:",
        f"{multiplier_without_pivoting:.6e}",
    )

    print(
        "Multiplier with partial pivoting:",
        f"{multiplier_with_pivoting:.6e}",
    )

    # Compare the two Gaussian elimination approaches.
    additional_solvers = {
        "Gaussian elimination (no pivoting)": (
            gaussian_solve_without_pivoting
        ),
    }

    run_solver_comparison(
        A,
        b,
        x_exact,
        "SOLVER COMPARISON",
        additional_solvers,
    )


if __name__ == "__main__":
    main()
