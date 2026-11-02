
"""Experiment 1: Solve a well-conditioned linear system."""

import numpy as np

from experiments.common import run_solver_comparison


def create_system():
    """Create a well-conditioned system with a known exact solution."""

    A = np.array([
        [4.0, 1.0, 1.0],
        [1.0, 3.0, 0.0],
        [1.0, 0.0, 2.0],
    ])

    x_exact = np.array([1.0, 2.0, 3.0])

    b = A @ x_exact

    return A, b, x_exact


def main():

    A, b, x_exact = create_system()

    run_solver_comparison(
        A,
        b,
        x_exact,
        "EXPERIMENT 1: WELL-CONDITIONED SYSTEM",
    )


if __name__ == "__main__":
    main()
