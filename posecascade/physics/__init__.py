"""PMX rigid-body physics + 6DOF spring joints.

Pure-Python semi-implicit Euler integration. Deterministic across
platforms (no LCP solver in the loop). Rigid-vs-rigid collision is
handled by an O(N²) AABB broadphase + per-pair narrowphase that
covers every shape combination PMX models actually ship
(sphere / capsule / box). PMX group + non_collision_mask filtering is
applied at broadphase time.
"""

from posecascade.physics.collision import (
    Contact,
    find_contacts,
    resolve_contacts,
)
from posecascade.physics.types import (
    Joint6DofSpring,
    PhysicsMode,
    PhysicsScene,
    RigidBody,
    RigidShape,
)
from posecascade.physics.world import PhysicsWorld

__all__ = [
    "Contact",
    "Joint6DofSpring",
    "PhysicsMode",
    "PhysicsScene",
    "PhysicsWorld",
    "RigidBody",
    "RigidShape",
    "find_contacts",
    "resolve_contacts",
]
