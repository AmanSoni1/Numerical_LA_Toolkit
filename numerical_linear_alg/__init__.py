"""Numerical Linear Algebra Toolkit."""

# Gaussian elimination
from .gaussian import gaussian_solve, back_substitution

# LU decomposition
from .lu import lu_factor, lu_solve

# Numerical diagnostics
from .diagnostics import (
    residual,
    relative_residual,
    backward_error,
    condition_number,
)

__all__ = [
    "gaussian_solve",
    "back_substitution",
    "lu_factor",
    "lu_solve",
    "residual",
    "relative_residual",
    "backward_error",
    "condition_number",
]