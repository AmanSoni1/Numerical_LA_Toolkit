# Numerical Linear Algebra Toolkit

An object-oriented Python package for solving linear systems **Ax = b** and for investigating the accuracy and reliability of numerical algorithms. The core algorithms (Gaussian elimination, LU factorization and the diagnostics) are implemented from scratch; NumPy is used only for basic array operations and for validating results.

Two-week collaborative scientific computing project — Python · NumPy · GitHub.

## Object-oriented design

Every algorithm is a class. Solvers share one abstract interface, so they can be used interchangeably (polymorphism); the LU factors live inside an object so they can be reused (encapsulation); the diagnostics extend a base class (inheritance).

```
LinearSolver (abstract base class)               numerical_linear_alg/base.py
├── GaussianEliminationSolver                    gaussian.py
├── LUSolver                                     lu.py
└── NumpySolver, GaussianEliminationNoPivoting   (experiments only)

LUDecomposition                                  lu.py
    stores P, L, U; reused for many right-hand sides

MatrixDiagnostics                                diagnostics.py
└── SolutionDiagnostics                          inherits the matrix diagnostics

Experiment (abstract base class)                 experiments/common.py
├── WellConditionedExperiment
├── PivotingExperiment
└── HilbertExperiment
SolverComparison                                 runs any set of solvers on one system
```

| Class | Role | Main methods |
|---|---|---|
| `LinearSolver` | Abstract interface shared by every solver; a solver object can also be called as `solver(A, b)`. | `solve(A, b)` |
| `GaussianEliminationSolver` | Gaussian elimination with partial pivoting; handles row swaps and reports singular pivots. | `solve(A, b)`, `back_substitution(U, b)` |
| `LUDecomposition` | Holds the factors of `PA = LU`; solves with `Ly = Pb`, `Ux = y`. | `from_matrix(A)`, `solve(b)`, `forward_substitution(b)`, `back_substitution(y)`, `factors` |
| `LUSolver` | Factorizes `A` with pivoting and then solves. | `solve(A, b)`, `factorize(A)` |
| `MatrixDiagnostics` | Properties of the matrix alone. | `condition_number()` |
| `SolutionDiagnostics` | Accuracy of a computed solution `x` of `Ax = b` (2-norms throughout). | `residual()`, `relative_residual()`, `backward_error()` |
| `SolverComparison` | Compares several solvers on one system with a known exact solution. | `run()` |
| `Experiment` | Abstract base class of the three experiments. | `run()` |

The diagnostics are:

- residual `r = b − Ax̂`
- relative residual `‖r‖₂ / ‖b‖₂`
- normwise relative backward error `‖r‖₂ / (‖A‖₂‖x̂‖₂ + ‖b‖₂)`
- 2-norm condition number

## Requirements

- Python 3.9 or newer
- `numpy`, `matplotlib`, `pytest` (listed in `requirements.txt`)

## Setup

```bash
git clone <repository-url>
cd Numerical_LA_Toolkit

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run all examples from the repository root so that the `numerical_linear_alg` package can be imported.

```python
import numpy as np
from numerical_linear_alg import (
    GaussianEliminationSolver, LUSolver, SolutionDiagnostics,
)

A = np.array([[4.0, 1.0, 1.0],
              [1.0, 3.0, 0.0],
              [1.0, 0.0, 2.0]])
b = A @ np.array([1.0, 2.0, 3.0])

# Any solver can be swapped for another: they share the same interface
x = GaussianEliminationSolver().solve(A, b)

# LU: factor once, reuse for many right-hand sides
lu = LUSolver().factorize(A)       # LUDecomposition object
x1 = lu.solve(b)
x2 = lu.solve(2 * b)
P, L, U = lu.factors               # P A = L U

# Diagnostics
d = SolutionDiagnostics(A, x, b)
print(d.relative_residual(), d.backward_error(), d.condition_number())
```

`A` and `b` are never modified; lists are accepted as well as NumPy arrays.

### Error handling

| Situation | Behaviour |
|---|---|
| Non-square `A`, or `b` of the wrong length | `ValueError` |
| NaN or infinite entries | `ValueError` |
| Singular matrix (pivot below `1e-14`) | `ValueError` |
| `b = 0` in `relative_residual()` | `ValueError` |

## Experiments

Each experiment compares Gaussian elimination, LU factorization and NumPy's `np.linalg.solve`, reports solution error, residual and backward error, and validates the package diagnostics against independent calculations.

| Command | Experiment class | Description |
|---|---|---|
| `python -m experiments.well_conditioned` | `WellConditionedExperiment` | **1.** Well-conditioned 3×3 system with a known exact solution. |
| `python -m experiments.pivoting` | `PivotingExperiment` | **2.** Why partial pivoting matters: `A = [[1e-12, 1], [1, 1]]`, `x_exact = [1, 1]`, compared with elimination without pivoting. |
| `python -m experiments.hilbert` | `HilbertExperiment` | **3.** Hilbert matrices for n = 3, 5, 8, 10: relative solution error against condition number. |

The Hilbert experiment writes `experiments/hilbert_results.txt` (table) and `experiments/hilbert_conditioning.png` (plot). On a machine without a display, run it with a non-interactive backend:

```bash
MPLBACKEND=Agg python -m experiments.hilbert     # Windows PowerShell: $env:MPLBACKEND="Agg"
```

## Running the tests

```bash
python -m pytest
```

The tests cover correctness against NumPy, `PA = LU` reconstruction, factorization reuse, singular and invalid inputs, input immutability, the diagnostics formulas, the class hierarchy and the experiment utilities.

## Reproducing everything from a clean environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
MPLBACKEND=Agg python -m experiments.well_conditioned
MPLBACKEND=Agg python -m experiments.pivoting
MPLBACKEND=Agg python -m experiments.hilbert
```

Last digits of the results can differ slightly between machines because of the underlying BLAS/LAPACK library.

## Project structure

```
numerical_linear_alg/
    __init__.py          public API
    base.py              LinearSolver abstract base class
    gaussian.py          GaussianEliminationSolver
    lu.py                LUDecomposition, LUSolver
    diagnostics.py       MatrixDiagnostics, SolutionDiagnostics
experiments/
    common.py            Experiment, SolverComparison, NumpySolver
    well_conditioned.py  experiment 1
    pivoting.py          experiment 2
    gaussian_no_pivoting.py   GaussianEliminationNoPivoting (experiments only)
    hilbert.py           experiment 3
tests/                   pytest test suite
CONTRIBUTIONS.md         roles, issues, pull requests and reviews
```

## Team

| Member | Responsibility |
|---|---|
| Aman Soni | Team leader, Gaussian elimination |
| Aditya Kumar | LU factorization |
| Aparna Singh | Numerical diagnostics |
| Anvesh Devulapalli | Experiments and validation |

See [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for details.

## Workflow

Work is tracked as GitHub issues, developed on feature branches and merged into `main` through pull requests that include tests and at least one teammate's review. There are no direct commits to `main`.