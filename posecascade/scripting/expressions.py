"""Tiny safe expression evaluator for declarative animation values.

Animation JSON sometimes needs an inline formula — e.g. driving a
bone amplitude by ``"0.5 * sin(elapsed * tau)"`` — without resorting
to a full Python script. This module compiles such expressions
through Python's ``ast`` module, walks the resulting tree to verify
every node is in a small whitelist (numeric literals, the binary /
unary operators, named variables drawn from the scope, and calls to
the curated math helpers below), then evaluates against a per-tick
scope of floats.

What's intentionally NOT supported:

* attribute access (``foo.bar``) — closes off the introspection paths
  ``eval`` would otherwise expose;
* subscripting / containers / comprehensions — no need, and they
  enlarge the attack surface;
* assignment, ``import``, ``lambda``, walrus, comparisons,
  list / dict / set literals — same;
* string operations — values must collapse to floats.

The evaluator is deterministic, allocation-light (one walk per call),
and raises :class:`ExpressionError` with the offending source on any
violation so authors get a readable message instead of a Python
traceback.
"""
from __future__ import annotations

import ast
import math
import operator as _op
from collections.abc import Callable
from typing import Any

from posecascade.errors import PoseCascadeError


class ExpressionError(PoseCascadeError):
    """Raised when an expression is malformed or uses a denied construct."""


_BINARY_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
    ast.Pow: _op.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: _op.pos,
    ast.USub: _op.neg,
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _sign(x: float) -> float:
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0


_FUNCTIONS: dict[str, Callable[..., float]] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "abs": abs,
    "min": min,
    "max": max,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "pow": math.pow,
    "clamp": _clamp,
    "lerp": _lerp,
    "sign": _sign,
}

_BUILTIN_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "tau": math.tau,
    "e": math.e,
    "inf": math.inf,
}


def evaluate_expression(source: str, scope: dict[str, float]) -> float:
    """Compile + evaluate ``source`` once against ``scope``.

    Caller is expected to have parsed ``source`` once at load time and
    cached the result, but for simplicity this function re-parses each
    call — animation expressions are short and per-frame parsing is
    cheap. ``scope`` provides per-frame variables (e.g. ``elapsed``,
    ``phase_t``); the math constants in :data:`_BUILTIN_CONSTANTS` are
    layered underneath so authors can reference ``tau`` etc. without
    redefining them.
    """
    if not isinstance(source, str):
        raise ExpressionError(f"expression must be str, got {type(source).__name__}")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as err:
        raise ExpressionError(f"syntax error in {source!r}: {err.msg}") from err
    merged: dict[str, float] = {**_BUILTIN_CONSTANTS, **scope}
    return float(_evaluate_node(tree.body, merged, source))


def _evaluate_node(node: ast.AST, scope: dict[str, float], source: str) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ExpressionError(
            f"constants in {source!r} must be numeric, got {type(node.value).__name__}",
        )
    if isinstance(node, ast.Name):
        name = node.id
        if name not in scope:
            raise ExpressionError(
                f"unknown identifier {name!r} in {source!r}; "
                f"available: {sorted(scope)}",
            )
        return float(scope[name])
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ExpressionError(
                f"unary op {op_type.__name__} not allowed in {source!r}",
            )
        return _UNARY_OPS[op_type](_evaluate_node(node.operand, scope, source))
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINARY_OPS:
            raise ExpressionError(
                f"binary op {op_type.__name__} not allowed in {source!r}",
            )
        left = _evaluate_node(node.left, scope, source)
        right = _evaluate_node(node.right, scope, source)
        return _BINARY_OPS[op_type](left, right)
    if isinstance(node, ast.Call):
        return _evaluate_call(node, scope, source)
    raise ExpressionError(
        f"AST node {type(node).__name__} not allowed in {source!r}",
    )


def _evaluate_call(node: ast.Call, scope: dict[str, float], source: str) -> float:
    if not isinstance(node.func, ast.Name):
        raise ExpressionError(f"only direct function calls allowed in {source!r}")
    name = node.func.id
    if name not in _FUNCTIONS:
        raise ExpressionError(
            f"function {name!r} not allowed in {source!r}; "
            f"available: {sorted(_FUNCTIONS)}",
        )
    if node.keywords:
        raise ExpressionError(
            f"keyword args not supported in {source!r} (function {name})",
        )
    args = [_evaluate_node(a, scope, source) for a in node.args]
    return float(_FUNCTIONS[name](*args))


def looks_like_expression(value: Any) -> bool:
    """Cheap heuristic — does the string contain non-symbolic operators?

    Plain symbolic constants like ``"pi"`` or ``"tau"`` are NOT treated
    as expressions (they go through the original symbolic-scalar
    fast path); only strings containing arithmetic, parens, or
    function-call syntax trigger the AST evaluator.
    """
    if not isinstance(value, str):
        return False
    return any(ch in value for ch in "+-*/%(),") or value.endswith(")")


__all__ = [
    "ExpressionError",
    "evaluate_expression",
    "looks_like_expression",
]
