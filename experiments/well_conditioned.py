
"""Experiment 1: Solve a well-conditioned linear system."""

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import Experiment, SolverComparison


class WellConditionedExperiment(Experiment):
    """Solve a well-conditioned system with a known exact solution."""

    title = "EXPERIMENT 1: WELL-CONDITIONED SYSTEM"

    def create_system(self):
        """Create a well-conditioned system with a known exact solution."""

        A = np.array([
            [4.0, 1.0, 1.0],
            [1.0, 3.0, 0.0],
            [1.0, 0.0, 2.0],
        ])

        x_exact = np.array([1.0, 2.0, 3.0])

        b = A @ x_exact

        return A, b, x_exact

    def run(self):
        A, b, x_exact = self.create_system()

        return SolverComparison(A, b, x_exact, self.title).run()


# Backward-compatible module-level API
def create_system():
    return WellConditionedExperiment().create_system()


def main():
    WellConditionedExperiment().main()


if __name__ == "__main__":
    main()
