"""Numerical Linear Algebra Toolkit (object-oriented)."""

# Common interface
from .base import LinearSolver

# Gaussian elimination
from .gaussian import GaussianEliminationSolver, gaussian_solve, back_substitution

# LU decomposition
from .lu import LUDecomposition, LUSolver, lu_factor, lu_solve

# Numerical diagnostics
from .diagnostics import (
    MatrixDiagnostics,
    SolutionDiagnostics,
    residual,
    relative_residual,
    backward_error,
    condition_number,
)

__all__ = [
    # classes
    "LinearSolver",
    "GaussianEliminationSolver",
    "LUDecomposition",
    "LUSolver",
    "MatrixDiagnostics",
    "SolutionDiagnostics",
    # functional API kept for backward compatibility
    "gaussian_solve",
    "back_substitution",
    "lu_factor",
    "lu_solve",
    "residual",
    "relative_residual",
    "backward_error",
    "condition_number",
]
