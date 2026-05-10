"""Built-in MMD-style pose presets for the declarative animation runtime.

Each preset maps ``bone_key → {x_rad?, y_rad?, z_rad?: float}``. Authors
reference a preset on a phase via ``pose: "name"`` (or
``pose: {name: "name", weight: curve}``) and the runtime composes it
under the phase's per-bone overrides — phase-level ``bones`` always wins
on a per-axis basis, so a preset gives a starting silhouette that the
phase can tweak without redeclaring everything.

Documents may declare their own ``pose_library: { name: { ... } }`` to
override the built-ins of the same name or to add new presets.

Bone keys go through the rig's ``body_bones`` alias map at runtime, so a
preset authored in Galaxia / VRoid vocabulary
(``upper_arm_L``, ``head``, ...) plays on any rig that aliases those
names. Magnitudes match dance.py's tested ranges so they read clearly
on the same character without proximity-weight pinching.
"""
from __future__ import annotations

PoseSpec = dict[str, dict[str, float]]


BUILTIN_POSES: dict[str, PoseSpec] = {
    # Both arms straight overhead in a V — the canonical MMD finale shape.
    "v_arms_up": {
        "upper_arm_L": {"x_rad": -1.30, "z_rad": 0.55},
        "upper_arm_R": {"x_rad": -1.30, "z_rad": -0.55},
        "head": {"x_rad": -0.18},
        "chest": {"x_rad": -0.10},
    },
    # Arms fold across the chest — a "cool intro" silhouette.
    "arms_to_chest": {
        "upper_arm_L": {"x_rad": -0.45, "z_rad": -0.50},
        "upper_arm_R": {"x_rad": -0.45, "z_rad": 0.50},
        "chest": {"x_rad": 0.10},
    },
    # Hip pop right (body weight on right foot, left hip out).
    "hip_pop_L": {
        "hip": {"y_rad": 0.30},
        "chest": {"y_rad": -0.18, "x_rad": 0.05},
        "head": {"y_rad": 0.20},
    },
    # Mirror of hip_pop_L.
    "hip_pop_R": {
        "hip": {"y_rad": -0.30},
        "chest": {"y_rad": 0.18, "x_rad": 0.05},
        "head": {"y_rad": -0.20},
    },
    # Right arm points / reaches up-right; head turns to follow.
    "point_R": {
        "upper_arm_R": {"x_rad": -1.10, "z_rad": -0.30},
        "upper_arm_L": {"x_rad": 0.10, "z_rad": 0.40},
        "head": {"y_rad": -0.20},
        "chest": {"y_rad": -0.05},
    },
    # Mirror of point_R.
    "point_L": {
        "upper_arm_L": {"x_rad": -1.10, "z_rad": 0.30},
        "upper_arm_R": {"x_rad": 0.10, "z_rad": -0.40},
        "head": {"y_rad": 0.20},
        "chest": {"y_rad": 0.05},
    },
    # Hands meet at the chest centre — useful for prayer / sing poses.
    "hands_clasp": {
        "upper_arm_L": {"x_rad": -0.40, "z_rad": -0.45},
        "upper_arm_R": {"x_rad": -0.40, "z_rad": 0.45},
        "chest": {"x_rad": 0.05},
    },
    # Soft reach to the right — R arm extends forward-right, body
    # turns + leans toward it, L arm relaxes slightly back. Shallower
    # forward angle than ``point_R`` so it reads as "casual reach"
    # not "stab", but z_rad pushes a full 0.55 outward (matching
    # ``v_arms_up``) so the hand swings WIDE OF the torso instead
    # of grazing the skirt — the previous 0.20 wasn't enough lateral
    # offset on the bundled VRoid rig and the hand clipped through.
    "reach_R_soft": {
        "upper_arm_R": {"x_rad": -0.85, "z_rad": -0.55},
        "upper_arm_L": {"x_rad": 0.05, "z_rad": 0.45},
        "head": {"y_rad": -0.15, "x_rad": -0.05},
        "chest": {"x_rad": -0.04, "y_rad": -0.05},
    },
    # Mirror of reach_R_soft.
    "reach_L_soft": {
        "upper_arm_L": {"x_rad": -0.85, "z_rad": 0.55},
        "upper_arm_R": {"x_rad": 0.05, "z_rad": -0.45},
        "head": {"y_rad": 0.15, "x_rad": -0.05},
        "chest": {"x_rad": -0.04, "y_rad": 0.05},
    },
    # Both arms angled outward + slightly raised — "open invitation"
    # silhouette. Stays clear of the torso because z_rad pulls each
    # arm away from the body centreline; less raise than v_arms_up so
    # the elbow stays roughly at shoulder height.
    "arms_open": {
        "upper_arm_L": {"x_rad": -0.60, "z_rad": 0.60},
        "upper_arm_R": {"x_rad": -0.60, "z_rad": -0.60},
        "head": {"x_rad": -0.08},
        "chest": {"x_rad": -0.06},
    },
}


def merge_libraries(user: dict[str, PoseSpec] | None) -> dict[str, PoseSpec]:
    """Overlay a user-supplied library on the built-ins.

    User entries with the same name as a built-in win wholesale (the
    preset is replaced, not field-merged) — same precedence as renaming
    a built-in pose. Returns a fresh dict so the caller can mutate
    safely without touching :data:`BUILTIN_POSES`.
    """
    merged: dict[str, PoseSpec] = {
        name: {bone: dict(axes) for bone, axes in spec.items()}
        for name, spec in BUILTIN_POSES.items()
    }
    if user:
        for name, spec in user.items():
            merged[str(name)] = {
                str(bone): {str(axis): float(v) for axis, v in axes.items()}
                for bone, axes in spec.items()
            }
    return merged


__all__ = ["BUILTIN_POSES", "PoseSpec", "merge_libraries"]
