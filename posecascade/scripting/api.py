"""Helper functions exposed to user scripts via the sandbox.

These are pure, side-effect-free utilities — vector / quaternion construction,
math primitives, easing helpers — that user scripts call without needing to
``import`` anything (which the sandbox forbids).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from posecascade.animation.ik import solve_two_bone
from posecascade.scene.node import Node
from posecascade.utils.math3d import (
    Quat,
    Vec3,
    normalize,
    quat_inverse,
    quat_mul,
    vec3,
)


def quat_axis_angle(axis: Vec3, angle_radians: float) -> Quat:
    """Build a unit quaternion from an axis (3-vec) and angle in radians."""
    unit_axis = normalize(np.asarray(axis, dtype=np.float32))
    half = float(angle_radians) * 0.5
    sin_half = math.sin(half)
    return np.array(
        [
            float(unit_axis[0]) * sin_half,
            float(unit_axis[1]) * sin_half,
            float(unit_axis[2]) * sin_half,
            math.cos(half),
        ],
        dtype=np.float32,
    )


def quat_from_euler(yaw: float, pitch: float, roll: float) -> Quat:
    """Tait-Bryan ZYX from yaw/pitch/roll (radians) → unit quaternion."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float32,
    )


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation; ``t`` is unclamped."""
    return float(a) + (float(b) - float(a)) * float(t)


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` into ``[lo, hi]``."""
    return max(lo, min(hi, value))


_VEC3_LEN = 3


def world_position(node: object) -> Vec3:
    """Walk the parent chain and return ``node``'s world-space origin.

    User scripts that drive IK targets need a way to read where a bone
    sits in world space (e.g. snapshot a foot's planted position so it
    stays pinned across body translation). The Node API only exposes
    local transforms; this helper does the chain composition.
    """
    if not isinstance(node, Node):
        raise TypeError(f"world_position expects a Node, got {type(node).__name__}")
    matrix = node.transform.to_matrix()
    parent = node.parent
    while parent is not None:
        matrix = parent.transform.to_matrix() @ matrix
        parent = parent.parent
    return np.array([matrix[0, 3], matrix[1, 3], matrix[2, 3]], dtype=np.float32)


class IkApi:
    """Curated IK surface — currently the 2-bone solver.

    Exposed in the script sandbox as ``ik``. Each call to
    :meth:`solve_two_bone` rotates ``root`` and ``mid`` so ``end``'s
    world origin lands at ``target`` — useful for foot pinning during
    walks / stair climbs (script computes a desired foot world
    position and lets the engine bend the knee to reach it).
    """

    def solve_two_bone(
        self,
        root: object,
        mid: object,
        end: object,
        target: object,
        iterations: int = 8,
        step_radian: float = 0.6,
        mid_limit_min: tuple[float, float, float] | None = None,
        mid_limit_max: tuple[float, float, float] | None = None,
    ) -> None:
        """Place ``end``'s world origin at ``target`` by rotating ``root`` and ``mid``.

        ``mid_limit_min`` / ``mid_limit_max`` (XYZ Euler radians) clamp
        the mid joint's local rotation each CCD step; pass them to
        constrain a knee or elbow to a hinge axis (e.g. knee bending
        only around X: ``mid_limit_min=(-2.4, 0.0, 0.0)``,
        ``mid_limit_max=(0.1, 0.0, 0.0)``).
        """
        if not isinstance(root, Node) or not isinstance(mid, Node) or not isinstance(end, Node):
            raise TypeError("ik.solve_two_bone expects three Node arguments")
        target_arr = np.asarray(target, dtype=np.float32).reshape(-1)
        if target_arr.shape[0] != _VEC3_LEN:
            raise ValueError(f"target must be a 3-vector, got shape {target_arr.shape}")
        solve_two_bone(
            root,
            mid,
            end,
            target_arr,
            iterations=iterations,
            step_radian=step_radian,
            mid_limit_min=mid_limit_min,
            mid_limit_max=mid_limit_max,
        )


_IK_SINGLETON = IkApi()


SCRIPT_API_HELPERS: dict[str, Any] = {
    "vec3": vec3,
    "quat_axis_angle": quat_axis_angle,
    "quat_from_euler": quat_from_euler,
    # ``quat_mul`` lets a script compose a yaw / spin on top of a node's
    # *export* rotation (e.g. the 180° axis-fix Sketchfab glTFs ship on
    # their root) instead of clobbering it with ``set_rotation``.
    "quat_mul": quat_mul,
    "quat_inverse": quat_inverse,
    "lerp": lerp,
    "clamp": clamp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "pi": math.pi,
    "tau": math.tau,
    "ik": _IK_SINGLETON,
    "world_position": world_position,
}
