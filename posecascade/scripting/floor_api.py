"""Sandbox-facing facade over :class:`~posecascade.animation.foot_planting.FootPlanter`.

Scripts can't import engine modules from inside the sandbox, so this
class wraps :class:`FootPlanter` in a minimal surface that takes
plain Node refs / floats / callables and produces ``FootIKChain``
records internally. Exposed in the script globals as ``floor``.
"""
from __future__ import annotations

from posecascade.animation.foot_planting import (
    FootIKChain,
    FootPlanter,
    GroundProvider,
    auto_foot_samples,
    flat_ground,
    stair_ground,
)
from posecascade.scene.node import Node


class FloorApi:
    """Curated foot-planting surface for sandboxed user scripts."""

    def __init__(self, planter: FootPlanter, skins=(), meshes=()) -> None:
        self._planter = planter
        self._skins = skins
        self._meshes = meshes

    def clear(self) -> None:
        """Forget every previously bound foot. Call this in ``start()``
        before re-binding so a script reload doesn't accumulate ghosts
        from the previous run."""
        self._planter.clear()

    def bind_foot(
        self,
        root: object,
        mid: object,
        end: object,
        ground: object,
        sole_samples: tuple | list | None = None,
        sole_down_local: tuple[float, float, float] | None = None,
        toe_forward_local: tuple[float, float, float] | None = None,
        knee_limit_min: tuple[float, float, float] | None = None,
        knee_limit_max: tuple[float, float, float] | None = None,
    ) -> None:
        """Register a 2-bone IK chain (root → mid → end) bound to ``ground``.

        ``ground`` may be a callable ``(x, z) -> y`` (in which case it
        is used directly) or a float (treated as a constant flat-ground
        elevation).

        ``sole_samples`` is an iterable of ``(local_axis, world_distance)``
        pairs describing the foot's contact surface relative to the
        foot bone joint. Each ``local_axis`` is a 3-tuple in the foot
        bone's local frame (a unit vector — the engine normalises and
        re-applies the world distance separately, so users don't have
        to compensate for rig import scales). For Galaxia-style rigs
        with the foot bone's local -Z pointing toward the sole and
        local +Y pointing toward the toes, a typical value is
        ``[((0, -0.2, -1), 0.020), ((0, +0.7, -1), 0.020)]`` — sampling
        heel and toe at 2 cm below the ankle joint. Default is a
        single sample at the joint origin (no offset), which matches
        the legacy "ankle = sole" assumption.

        Knee limits, when set, only affect the script-side IK; the
        planter's corrective solve ignores them so it can converge
        on heavily yawed bodies.
        """
        if not isinstance(root, Node) or not isinstance(mid, Node) or not isinstance(end, Node):
            raise TypeError("floor.bind_foot expects three Node arguments (root, mid, end)")
        provider = _coerce_ground(ground)
        if sole_samples is None and self._skins and self._meshes:
            # No explicit samples → auto-derive from the skinned mesh.
            # The engine walks dominant vertices, picks heel / sole-
            # centre / toe extremes, and sets ``foot_offset`` to a
            # numerical safety margin. Scripts only need to nominate
            # the foot bone and ground provider.
            auto_samples, auto_offset = auto_foot_samples(
                end, self._skins, self._meshes,
            )
            if auto_samples:
                coerced_samples = auto_samples
                self._planter.foot_offset = max(self._planter.foot_offset, auto_offset)
            else:
                coerced_samples = _coerce_sole_samples(None)
        else:
            coerced_samples = _coerce_sole_samples(sole_samples)
        coerced_sole_down = (
            tuple(float(c) for c in sole_down_local)
            if sole_down_local is not None
            else None
        )
        coerced_toe_forward = (
            tuple(float(c) for c in toe_forward_local)
            if toe_forward_local is not None
            else None
        )
        chain = FootIKChain(
            root=root,
            mid=mid,
            end=end,
            ground=provider,
            sole_samples=coerced_samples,
            sole_down_local=coerced_sole_down,
            toe_forward_local=coerced_toe_forward,
            knee_limit_min=knee_limit_min,
            knee_limit_max=knee_limit_max,
        )
        self._planter.bind(chain)

    def set_body_forward(self, direction: tuple[float, float, float]) -> None:
        """Update the body's forward direction in world coordinates.

        Call this every frame after the script computes its current
        body yaw. The planter uses it as the secondary aim target
        when aligning the foot's toe-forward axis after lift; without
        it the foot is flat-on-floor but its toes can end up pointing
        sideways or backward as the body yaw changes phase by phase.
        """
        if len(direction) != _SOLE_AXIS_LEN:
            raise ValueError(
                f"body forward direction must have 3 components, got {direction!r}",
            )
        self._planter.body_forward_world = tuple(float(c) for c in direction)

    def set_foot_offset(self, offset: float) -> None:
        """Distance from the foot bone's joint origin down to the sole.

        IK targets are placed at ``ground + offset`` so the rest pose's
        foot bone (whose head joint typically sits a few cm above the
        sole) lands ON the floor instead of sinking into it. Tune this
        once per character based on the rig's foot bone height.
        """
        self._planter.foot_offset = float(offset)

    @staticmethod
    def flat(y: float = 0.0) -> GroundProvider:
        """A constant-elevation provider for level floors."""
        return flat_ground(y=float(y))

    @staticmethod
    def stairs(
        *,
        base_z: float,
        step_depth: float,
        step_rise: float,
        count: int,
        base_y: float = 0.0,
        forward_sign: int = -1,
        edge_smooth: float | None = None,
    ) -> GroundProvider:
        """A staircase elevation field — see :func:`stair_ground` for params."""
        return stair_ground(
            base_z=float(base_z),
            step_depth=float(step_depth),
            step_rise=float(step_rise),
            count=int(count),
            base_y=float(base_y),
            forward_sign=int(forward_sign),
            edge_smooth=None if edge_smooth is None else float(edge_smooth),
        )


def _coerce_ground(value: object) -> GroundProvider:
    if callable(value):
        return value  # type: ignore[return-value]
    if isinstance(value, (int, float)):
        return flat_ground(y=float(value))
    raise TypeError(
        f"ground must be a callable (x, z)->y or a float, got {type(value).__name__}"
    )


_SOLE_AXIS_LEN = 3


def _coerce_sole_samples(value):
    if value is None:
        return ((( 0.0, 0.0, 0.0), 0.0),)
    samples = []
    for entry in value:
        axis, distance = entry
        if len(axis) != _SOLE_AXIS_LEN:
            raise ValueError(f"sole sample axis must have 3 components, got {axis!r}")
        samples.append(
            (tuple(float(c) for c in axis), float(distance)),
        )
    if not samples:
        raise ValueError("sole_samples must contain at least one entry")
    return tuple(samples)


__all__ = ["FloorApi"]
