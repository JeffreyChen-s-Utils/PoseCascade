"""Tests for the safe expression evaluator used by declarative animations.

Pins both the happy path (numeric / variables / math fn calls) and the
guard rails (denied AST nodes raise :class:`ExpressionError` with a
readable message rather than triggering Python's ``eval``-style
introspection).
"""
from __future__ import annotations

import math

import pytest

from posecascade.scripting.expressions import (
    ExpressionError,
    evaluate_expression,
    looks_like_expression,
)

# --- Happy path -------------------------------------------------------------


def test_evaluate_numeric_literal() -> None:
    assert evaluate_expression("3.5", {}) == pytest.approx(3.5)


def test_evaluate_named_variable_from_scope() -> None:
    assert evaluate_expression("x + 1", {"x": 4.0}) == pytest.approx(5.0)


def test_evaluate_uses_builtin_math_constants() -> None:
    """``pi`` / ``tau`` / ``e`` are pre-loaded so authors can use them
    without redefining in scope."""
    assert evaluate_expression("tau / 2", {}) == pytest.approx(math.pi)
    assert evaluate_expression("e", {}) == pytest.approx(math.e)


def test_evaluate_arithmetic_with_precedence() -> None:
    assert evaluate_expression("2 + 3 * 4", {}) == pytest.approx(14.0)
    assert evaluate_expression("(2 + 3) * 4", {}) == pytest.approx(20.0)
    assert evaluate_expression("2 ** 3", {}) == pytest.approx(8.0)


def test_evaluate_unary_negation() -> None:
    assert evaluate_expression("-5", {}) == pytest.approx(-5.0)
    assert evaluate_expression("-(-3)", {}) == pytest.approx(3.0)


def test_evaluate_math_function_calls() -> None:
    """Whitelisted math fns (sin/cos/sqrt/clamp/lerp/...) work."""
    assert evaluate_expression("sin(pi / 2)", {}) == pytest.approx(1.0)
    assert evaluate_expression("cos(0)", {}) == pytest.approx(1.0)
    assert evaluate_expression("sqrt(16)", {}) == pytest.approx(4.0)
    assert evaluate_expression("clamp(5, 0, 1)", {}) == pytest.approx(1.0)
    assert evaluate_expression("clamp(-3, 0, 1)", {}) == pytest.approx(0.0)
    assert evaluate_expression("lerp(0, 10, 0.25)", {}) == pytest.approx(2.5)
    assert evaluate_expression("min(3, 7, 1)", {}) == pytest.approx(1.0)
    assert evaluate_expression("max(3, 7, 1)", {}) == pytest.approx(7.0)


def test_evaluate_animation_idiom() -> None:
    """The kind of formula the declarative runtime expects to see —
    sinusoidal oscillator using elapsed + tau."""
    scope = {"elapsed": 0.25, "amplitude": 0.5}
    # 0.5 * sin(0.25 * tau) = 0.5 * sin(π/2) = 0.5
    assert evaluate_expression(
        "amplitude * sin(elapsed * tau)", scope,
    ) == pytest.approx(0.5)


# --- Guard rails ------------------------------------------------------------


def test_unknown_identifier_raises() -> None:
    with pytest.raises(ExpressionError, match="unknown identifier"):
        evaluate_expression("ghost + 1", {"x": 0.0})


def test_undeclared_function_rejected() -> None:
    with pytest.raises(ExpressionError, match="not allowed"):
        evaluate_expression("evil(0)", {})


def test_attribute_access_rejected() -> None:
    """Closes off the ``__class__.__bases__`` introspection trick."""
    with pytest.raises(ExpressionError, match="not allowed"):
        evaluate_expression("x.__class__", {"x": 1.0})


def test_subscript_rejected() -> None:
    with pytest.raises(ExpressionError, match="not allowed"):
        evaluate_expression("x[0]", {"x": 1.0})


def test_lambda_rejected() -> None:
    with pytest.raises(ExpressionError, match="not allowed"):
        evaluate_expression("(lambda: 1)()", {})


def test_string_literal_rejected() -> None:
    with pytest.raises(ExpressionError, match="must be numeric"):
        evaluate_expression("'hello'", {})


def test_syntax_error_wraps_message() -> None:
    with pytest.raises(ExpressionError, match="syntax error"):
        evaluate_expression("3 +", {})


def test_keyword_args_rejected() -> None:
    with pytest.raises(ExpressionError, match="keyword args"):
        evaluate_expression("min(x=1)", {})


# --- Heuristic --------------------------------------------------------------


def test_looks_like_expression_picks_up_operators() -> None:
    assert looks_like_expression("a + b")
    assert looks_like_expression("sin(x)")
    assert looks_like_expression("x * 2")


def test_looks_like_expression_skips_plain_constants() -> None:
    """Symbolic constants like ``"pi"`` / ``"tau"`` should NOT be treated
    as expressions — they go through the legacy symbolic-scalar path."""
    assert not looks_like_expression("pi")
    assert not looks_like_expression("tau")
    assert not looks_like_expression("3.14")
    assert not looks_like_expression(3.14)  # type: ignore[arg-type]
