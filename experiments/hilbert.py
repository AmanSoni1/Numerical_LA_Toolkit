
"""Experiment 3: Investigate the numerical behaviour of Hilbert matrices."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from experiments.common import Experiment, SolverComparison


class HilbertExperiment(Experiment):
    """Study how accuracy degrades as the Hilbert matrix gets larger."""

    title = "EXPERIMENT 3: HILBERT MATRIX"

    SOLVER_NAMES = (
        SolverComparison.GAUSSIAN,
        SolverComparison.LU,
        SolverComparison.NUMPY,
    )

    def __init__(self, sizes=(3, 5, 8, 10), output_dir=None):
        self.sizes = list(sizes)
        self.output_dir = (
            Path(output_dir) if output_dir is not None
            else Path(__file__).resolve().parent
        )

        # Filled in by run().
        self.condition_numbers = []
        self.errors = {name: [] for name in self.SOLVER_NAMES}
        self.table_rows = []

    @staticmethod
    def create_hilbert_system(n):
        """Create a Hilbert matrix system with a known exact solution."""

        H = np.fromfunction(
            lambda i, j: 1.0 / (i + j + 1),
            (n, n),
            dtype=float,
        )

        x_exact = np.ones(n)

        b = H @ x_exact

        return H, b, x_exact

    # ------------------------------------------------------------------
    # Steps of the experiment
    # ------------------------------------------------------------------
    def _collect_results(self):
        """Solve every Hilbert system and gather the numerical results."""

        for n in self.sizes:

            H, b, x_exact = self.create_hilbert_system(n)

            results, kappa = SolverComparison(
                H,
                b,
                x_exact,
                f"{self.title} (n = {n})",
            ).run()

            self.condition_numbers.append(kappa)

            for solver_name in self.errors:

                metrics = results[solver_name]

                self.errors[solver_name].append(
                    metrics["solution_error"] / np.linalg.norm(x_exact)
                )

                self.table_rows.append(
                    (
                        n,
                        solver_name,
                        kappa,
                        metrics["solution_error"] / np.linalg.norm(x_exact),
                        metrics["relative_residual"],
                        metrics["backward_error"],
                    )
                )

    def _write_table(self):
        """Generate a text file containing the numerical results."""

        table_path = self.output_dir / "hilbert_results.txt"

        with open(table_path, "w", encoding="utf-8") as file:

            file.write("HILBERT MATRIX: NUMERICAL RESULTS\n")
            file.write("=" * 115 + "\n\n")

            file.write(
                f"{'n':<5}"
                f"{'Solver':<43}"
                f"{'Condition Number':>18}"
                f"{'Relative Error':>18}"
                f"{'Relative Residual':>18}"
                f"{'Backward Error':>18}\n"
            )

            file.write("-" * 120 + "\n")

            for n, solver, kappa, error, residual, backward in self.table_rows:

                file.write(
                    f"{n:<5}"
                    f"{solver:<43}"
                    f"{kappa:>18.6e}"
                    f"{error:>18.6e}"
                    f"{residual:>18.6e}"
                    f"{backward:>18.6e}\n"
                )

        print("\nNumerical results saved to:", table_path)

    def _plot(self):
        """Plot relative solution error against condition number."""

        plt.figure(figsize=(8, 5))

        for solver_name, solver_errors in self.errors.items():

            plt.loglog(
                self.condition_numbers,
                solver_errors,
                marker="o",
                label=solver_name,
            )

        plt.xlabel("Condition number")
        plt.ylabel("Relative solution error")
        plt.title("Hilbert Matrix: Solution Error vs Condition Number")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()

        plot_path = self.output_dir / "hilbert_conditioning.png"

        plt.savefig(plot_path, dpi=300)
        plt.show()

        print("\nPlot saved to:", plot_path)

    def run(self):
        self._collect_results()
        self._write_table()
        self._plot()


# Backward-compatible module-level API
def create_hilbert_system(n):
    return HilbertExperiment.create_hilbert_system(n)


def main():
    HilbertExperiment.main()


if __name__ == "__main__":
    main()
