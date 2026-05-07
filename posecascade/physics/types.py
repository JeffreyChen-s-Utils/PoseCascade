"""Engine-side rigid-body / joint dataclasses.

Mirrors PMX semantics tightly:

- :class:`RigidBody`'s ``physics_mode`` decides how it interacts with the
  bone the importer attached it to:

  - :attr:`PhysicsMode.KINEMATIC` — body follows the bone every step,
    ignoring forces. Use for body-collision proxies (chest, head).
  - :attr:`PhysicsMode.DYNAMIC` — bone follows the body. Use for free-
    floating accessories.
  - :attr:`PhysicsMode.DYNAMIC_BONE` — body integrates freely but its
    *position* is corrected back to the bone each step; only the
    rotation drives the bone. PMX's standard mode for hair / skirt.

- :class:`Joint6DofSpring` is the only joint type Phase 7 simulates.
  PMX's other joint types are parsed and stored verbatim for future
  phases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class RigidShape(IntEnum):
    SPHERE = 0
    BOX = 1
    CAPSULE = 2


class PhysicsMode(IntEnum):
    KINEMATIC = 0      # follows the bone; not pushed by forces
    DYNAMIC = 1        # free integration; bone follows the body
    DYNAMIC_BONE = 2   # integrate but snap position back to bone (rotation drives bone)


@dataclass(frozen=True)
class RigidBody:
    """One PMX rigid body.

    ``size`` semantics depend on ``shape``:

    - sphere: ``size[0]`` is the radius
    - box: ``size`` is the half-extent (x, y, z)
    - capsule: ``size[0]`` radius, ``size[1]`` height
    """

    name: str
    bone_index: int                                 # -1 = no bone attachment
    group: int                                      # 0..15
    # 16-bit field; ``mask & (1 << other_group)`` skips the pair.
    non_collision_mask: int
    shape: RigidShape
    size: tuple[float, float, float]
    position: tuple[float, float, float]            # initial (rest) world position
    rotation: tuple[float, float, float]            # initial Euler XYZ radians
    mass: float
    linear_damping: float
    angular_damping: float
    restitution: float
    friction: float
    physics_mode: PhysicsMode


@dataclass(frozen=True)
class Joint6DofSpring:
    """One PMX 6DOF spring joint.

    Each axis (x/y/z linear, x/y/z angular) carries an inclusive
    ``[lower, upper]`` range — exceeding it triggers a spring restoring
    force whose stiffness is the corresponding entry in
    ``spring_linear`` / ``spring_angular``.
    """

    name: str
    rigid_a_index: int
    rigid_b_index: int
    position: tuple[float, float, float]            # world-space anchor
    rotation: tuple[float, float, float]            # constraint frame Euler XYZ radians
    linear_lower: tuple[float, float, float]
    linear_upper: tuple[float, float, float]
    angular_lower: tuple[float, float, float]
    angular_upper: tuple[float, float, float]
    spring_linear: tuple[float, float, float]
    spring_angular: tuple[float, float, float]


@dataclass(frozen=True)
class PhysicsScene:
    """The bundle the importer hands to :class:`PhysicsWorld`."""

    bodies: tuple[RigidBody, ...] = field(default_factory=tuple)
    joints: tuple[Joint6DofSpring, ...] = field(default_factory=tuple)
