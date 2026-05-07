"""VMD bezier interpolation.

Each VMD bone keyframe carries four ``(X1, Y1, X2, Y2)`` control points
(uint8, range 0–127) that describe how the curve eases the X / Y / Z
position components and the rotation through that segment. To evaluate
the curve at a fraction ``t ∈ [0, 1]`` we treat the four control points
as a 2-D Bezier:

    P(s) = (1−s)³·(0, 0) + 3(1−s)²s·(X1, Y1) + 3(1−s)s²·(X2, Y2) + s³·(127, 127)

We solve ``P_x(s) = 127·t`` for the parameter ``s``, then return
``P_y(s) / 127`` as the eased fraction. Newton's method converges in
≤ 4 iterations for the 0..127 range; the bisection fallback guards
against tangent-near-zero edge cases (linear / step curves).
"""
from __future__ import annotations

import numpy as np

_CONTROL_MAX = 127.0
_NEWTON_ITERATIONS = 5
_BISECTION_ITERATIONS = 12
_TANGENT_EPSILON = 1.0e-6


def evaluate_bezier(handles: tuple[int, int, int, int], t: float) -> float:
    """Return the eased ``y`` value at parameter ``t`` (clamped to ``[0, 1]``).

    ``handles`` are the four 0–127 control values ``(X1, Y1, X2, Y2)`` in
    VMD's native byte form. ``t = 0`` and ``t = 1`` short-circuit so a
    string of identity-shaped keyframes never invokes the solver.
    """
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    x1 = handles[0] / _CONTROL_MAX
    y1 = handles[1] / _CONTROL_MAX
    x2 = handles[2] / _CONTROL_MAX
    y2 = handles[3] / _CONTROL_MAX
    if x1 == y1 and x2 == y2:
        # Linear curve: x(s) = s, so eased y == t.
        return t
    s = _solve_for_s(t, x1, x2)
    return _bezier_axis(s, y1, y2)


def evaluate_bezier_array(
    handles: tuple[int, int, int, int], ts: np.ndarray,
) -> np.ndarray:
    """Vectorised :func:`evaluate_bezier` for a numpy array of ``t`` values."""
    out = np.empty_like(ts, dtype=np.float32)
    for i in range(ts.size):
        out.flat[i] = evaluate_bezier(handles, float(ts.flat[i]))
    return out


def _bezier_axis(s: float, c1: float, c2: float) -> float:
    """Evaluate one axis of the 2-D bezier at parameter ``s``.

    Both axes share the same ``(0, 1)`` endpoint and the formula reduces
    to the standard cubic ``3(1−s)²s·c1 + 3(1−s)s²·c2 + s³`` once the
    endpoint terms collapse.
    """
    one_minus = 1.0 - s
    return 3.0 * one_minus * one_minus * s * c1 + 3.0 * one_minus * s * s * c2 + s * s * s


def _bezier_axis_derivative(s: float, c1: float, c2: float) -> float:
    """Derivative of :func:`_bezier_axis` w.r.t. ``s`` (used by Newton)."""
    one_minus = 1.0 - s
    return (
        3.0 * one_minus * one_minus * c1
        + 6.0 * one_minus * s * (c2 - c1)
        + 3.0 * s * s * (1.0 - c2)
    )


def _solve_for_s(target_x: float, x1: float, x2: float) -> float:
    """Solve ``bezier_x(s) = target_x`` for ``s`` using Newton + bisection."""
    s = target_x
    for _ in range(_NEWTON_ITERATIONS):
        x_at_s = _bezier_axis(s, x1, x2)
        derivative = _bezier_axis_derivative(s, x1, x2)
        if abs(derivative) < _TANGENT_EPSILON:
            break
        next_s = s - (x_at_s - target_x) / derivative
        if next_s < 0.0 or next_s > 1.0:
            break
        s = next_s
        if abs(x_at_s - target_x) < _TANGENT_EPSILON:
            return s
    return _bisect_for_s(target_x, x1, x2, s)


def _bisect_for_s(target_x: float, x1: float, x2: float, initial: float) -> float:
    """Bisection fallback when Newton's tangent collapses."""
    lo, hi = 0.0, 1.0
    s = initial
    for _ in range(_BISECTION_ITERATIONS):
        x_at_s = _bezier_axis(s, x1, x2)
        if abs(x_at_s - target_x) < _TANGENT_EPSILON:
            return s
        if x_at_s < target_x:
            lo = s
        else:
            hi = s
        s = (lo + hi) * 0.5
    return s
