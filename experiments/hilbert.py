
"""Experiment 3: Investigate the numerical behaviour of Hilbert matrices."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from experiments.common import run_solver_comparison


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


def main():

    sizes = [3, 5, 8, 10]

    condition_numbers = []

    errors = {
        "Gaussian elimination (partial pivoting)": [],
        "LU factorization (partial pivoting)": [],
        "NumPy": [],
    }

    # Collect numerical results for the summary table.
    table_rows = []

    for n in sizes:

        H, b, x_exact = create_hilbert_system(n)

        results, kappa = run_solver_comparison(
            H,
            b,
            x_exact,
            f"EXPERIMENT 3: HILBERT MATRIX (n = {n})",
        )

        condition_numbers.append(kappa)

        for solver_name in errors:

            metrics = results[solver_name]

            errors[solver_name].append(
                metrics["solution_error"] / np.linalg.norm(x_exact)
            )

            table_rows.append(
                (
                    n,
                    solver_name,
                    kappa,
                    metrics["solution_error"] / np.linalg.norm(x_exact),
                    metrics["relative_residual"],
                    metrics["backward_error"],
                )
            )

    # Generate a text file containing the numerical results.
    output_dir = Path(__file__).resolve().parent
    table_path = output_dir / "hilbert_results.txt"

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

        for n, solver, kappa, error, residual, backward in table_rows:

            file.write(
                f"{n:<5}"
                f"{solver:<43}"
                f"{kappa:>18.6e}"
                f"{error:>18.6e}"
                f"{residual:>18.6e}"
                f"{backward:>18.6e}\n"
            )

    print("\nNumerical results saved to:", table_path)

    # Plot relative solution error against condition number.
    plt.figure(figsize=(8, 5))

    for solver_name, solver_errors in errors.items():

        plt.loglog(
            condition_numbers,
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

    plot_path = output_dir / "hilbert_conditioning.png"

    plt.savefig(plot_path, dpi=300)
    plt.show()

    print("\nPlot saved to:", plot_path)


if __name__ == "__main__":
    main()
