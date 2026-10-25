"""Numerical Linear Algebra Toolkit: solving Ax = b and checking accuracy."""

""" it lets anyone write from numerical_linalg import gaussian_solve instead of from numerical_linalg.gaussian import gaussian_solve. __all__ just lists the public names."""
from .gaussian import gaussian_solve
from .lu import lu_factor, lu_solve
from .diagnostics import (
    residual,
    relative_residual,
    backward_error,
    condition_number,
)

__all__ = [
    "gaussian_solve",
    "lu_factor",
    "lu_solve",
    "residual",
    "relative_residual",
    "backward_error",
    "condition_number",
]