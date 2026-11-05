
"""Experiment 2: Investigate the importance of partial pivoting.

Compare Gaussian elimination with and without partial pivoting
using a linear system with a very small initial pivot.
"""

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import Experiment, SolverComparison
from experiments.gaussian_no_pivoting import GaussianEliminationNoPivoting


class PivotingExperiment(Experiment):
    """Show why partial pivoting matters when the first pivot is tiny."""

    title = "EXPERIMENT 2: IMPORTANCE OF PARTIAL PIVOTING"

    def create_system(self):
        """Create the linear system specified for the pivoting experiment."""

        A = np.array([
            [1e-12, 1.0],
            [1.0, 1.0],
        ])

        x_exact = np.array([1.0, 1.0])

        b = A @ x_exact

        return A, b, x_exact

    def run(self):
        A, b, x_exact = self.create_system()

        # Calculate the elimination multipliers.
        multiplier_without_pivoting = A[1, 0] / A[0, 0]
        multiplier_with_pivoting = A[0, 0] / A[1, 0]

        print(self.title)
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
        no_pivoting = GaussianEliminationNoPivoting()
        additional_solvers = {no_pivoting.name: no_pivoting}

        return SolverComparison(
            A,
            b,
            x_exact,
            "SOLVER COMPARISON",
            additional_solvers,
        ).run()


# Backward-compatible module-level API
def create_system():
    return PivotingExperiment().create_system()


def main():
    PivotingExperiment().main()


if __name__ == "__main__":
    main()
