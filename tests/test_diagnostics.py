import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from numerical_linear_alg.diagnostics import (
    backward_error,
    condition_number,
    relative_residual,
    residual,
)



A = np.array([[2.0, 1.0], [1.0, 3.0]])
X_EXACT = np.array([1.0, 2.0])
B = A @ X_EXACT

#Test cases
def test_residual_zero_for_exact_solution():
    assert np.allclose(residual(A, X_EXACT, B), 0.0)


def test_residual_known_value_by_hand():
    # x_hat = [0, 0]  ->  r = b - A*0 = b = [4, 7]
    assert np.allclose(residual(A, np.zeros(2), B), [4.0, 7.0])


def test_relative_residual_known_value():
    # x_hat = 0 -> r = b -> ||r||/||b|| = 1
    assert relative_residual(A, np.zeros(2), B) == pytest.approx(1.0)


def test_relative_residual_matches_formula():
    x_hat = np.array([1.1, 1.9])
    r = B - A @ x_hat
    expected = np.linalg.norm(r) / np.linalg.norm(B)
    assert relative_residual(A, x_hat, B) == pytest.approx(expected)


def test_relative_residual_b_zero_raises():
    with pytest.raises(ValueError, match="b = 0"):
        relative_residual(A, np.ones(2), np.zeros(2))


def test_backward_error_zero_for_exact_solution():
    assert backward_error(A, X_EXACT, B) == pytest.approx(0.0, abs=1e-15)


def test_backward_error_matches_formula():
    x_hat = np.array([1.1, 1.9])
    r = B - A @ x_hat
    expected = np.linalg.norm(r) / (
        np.linalg.norm(A, 2) * np.linalg.norm(x_hat) + np.linalg.norm(B)
    )
    assert backward_error(A, x_hat, B) == pytest.approx(expected)


def test_backward_error_zero_denominator_raises():
    with pytest.raises(ValueError):
        backward_error(np.zeros((2, 2)), np.zeros(2), np.zeros(2))


def test_condition_number_identity_is_one():
    assert condition_number(np.eye(3)) == pytest.approx(1.0)


def test_condition_number_diagonal():
    assert condition_number(np.diag([1.0, 100.0])) == pytest.approx(100.0)


def test_condition_number_singular_is_huge_or_inf():
    assert condition_number(np.array([[1.0, 2.0], [2.0, 4.0]])) > 1e15


def test_condition_number_hilbert_3():
    H = 1.0 / (np.arange(1, 4)[:, None] + np.arange(1, 4)[None, :] - 1)
    assert condition_number(H) == pytest.approx(524.0568, rel=1e-4)


def test_shape_errors():
    with pytest.raises(ValueError):
        residual(np.ones((2, 3)), np.ones(3), np.ones(2))
    with pytest.raises(ValueError):
        residual(A, np.ones(3), B)
    with pytest.raises(ValueError):
        condition_number(np.ones((2, 3)))git


def test_accepts_lists():
    assert np.allclose(residual([[2, 1], [1, 3]], [1, 2], [4, 7]), 0.0)
