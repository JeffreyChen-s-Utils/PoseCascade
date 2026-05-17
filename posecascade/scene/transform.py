"""TRS transform with cached world matrix.

A :class:`Transform` stores translation, rotation (unit quaternion), and scale.
World matrices are computed lazily; mutating any TRS component bumps a version
counter that the scene-graph traversal uses to invalidate cached children.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from posecascade.utils.math3d import (
    Mat4,
    Quat,
    Vec3,
    compose_trs,
    decompose_trs,
    mat4_identity,
    quat_identity,
    vec3,
)


@dataclass
class Transform:
    """Translation + rotation (quat) + scale, with a version counter for caching.

    Optionally carries a ``matrix_override`` 4x4 used verbatim by
    :meth:`to_matrix`; this lets the glTF importer keep raw bone matrices
    intact instead of decomposing them through :func:`decompose_trs`, which
    fails on the skewed/non-Euclidean matrices typical of skeleton joints
    and collapses the whole sub-tree to a near-zero-scale point. Mutating
    any TRS component clears the override so subsequent script-driven
    edits behave like a normal TRS transform again.
    """

    translation: Vec3 = field(default_factory=lambda: vec3(0.0, 0.0, 0.0))
    rotation: Quat = field(default_factory=quat_identity)
    scale: Vec3 = field(default_factory=lambda: vec3(1.0, 1.0, 1.0))
    version: int = 0
    matrix_override: Mat4 | None = field(default=None, repr=False, compare=False)

    def set_translation(self, value: Vec3) -> None:
        self._hydrate_trs_from_override()
        self.translation = value.astype(np.float32)
        self.matrix_override = None
        self.version += 1

    def set_rotation(self, value: Quat) -> None:
        self._hydrate_trs_from_override()
        self.rotation = value.astype(np.float32)
        self.matrix_override = None
        self.version += 1

    def set_scale(self, value: Vec3) -> None:
        self._hydrate_trs_from_override()
        self.scale = value.astype(np.float32)
        self.matrix_override = None
        self.version += 1

    def _hydrate_trs_from_override(self) -> None:
        """If ``matrix_override`` is set, decompose it into T/R/S BEFORE the
        caller mutates one of the three fields and the setter clears the
        override.

        glTF nodes with a ``matrix`` field land in ``from_raw_matrix`` and
        keep the bind transform exclusively in ``matrix_override``; the
        per-field translation / rotation / scale stay at constructor
        defaults. Without this hydration, a script (or the declarative
        runtime) that calls ``set_rotation(q)`` would clear the override,
        then ``to_matrix()`` falls back to ``compose_trs((0,0,0), q,
        (1,1,1))`` — losing the bone's bind translation entirely and
        teleporting it to its parent's origin.

        Hydrating reads the override's T/R/S once, copies them onto the
        TRS fields, then leaves it to the caller's setter to overwrite the
        specific one being changed. The remaining two preserve the bind.

        Skew in the override (rare in glTF, common in some FBX paths)
        round-trips lossily through TRS — same caveat as ``from_matrix``,
        documented there. The vast majority of glTF bones don't carry
        skew, so this is a clean fix for the typical case.
        """
        if self.matrix_override is None:
            return
        translation, rotation, scale = decompose_trs(self.matrix_override)
        self.translation = translation
        self.rotation = rotation
        self.scale = scale

    def to_matrix(self) -> Mat4:
        """Return the local transform matrix.

        Uses the raw matrix override if one is set; otherwise composes from TRS.
        """
        if self.matrix_override is not None:
            return self.matrix_override
        return compose_trs(self.translation, self.rotation, self.scale)

    @classmethod
    def from_matrix(cls, matrix: Mat4) -> Transform:
        """Decompose ``matrix`` into a TRS transform.

        Use :meth:`from_raw_matrix` instead when the matrix may be skewed or
        non-Euclidean (e.g. glTF skeleton bones) — decomposition is lossy for
        those and can produce a near-zero scale.
        """
        translation, rotation, scale = decompose_trs(matrix)
        return cls(translation=translation, rotation=rotation, scale=scale)

    @classmethod
    def from_raw_matrix(cls, matrix: Mat4) -> Transform:
        """Store ``matrix`` verbatim — :meth:`to_matrix` returns it as-is.

        Required for glTF nodes that ship a ``matrix`` array of bone-style
        values that ``decompose_trs`` cannot round-trip.
        """
        return cls(matrix_override=np.asarray(matrix, dtype=np.float32).copy())

    @classmethod
    def identity(cls) -> Transform:
        """Return the identity transform."""
        out = cls()
        # Force matrix realisation path via numpy so callers can sanity check.
        _ = mat4_identity()
        return out

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "translation": [float(v) for v in self.translation],
            "rotation": [float(v) for v in self.rotation],
            "scale": [float(v) for v in self.scale],
        }

    @classmethod
    def from_dict(cls, data: dict[str, list[float]]) -> Transform:
        return cls(
            translation=np.asarray(data["translation"], dtype=np.float32),
            rotation=np.asarray(data["rotation"], dtype=np.float32),
            scale=np.asarray(data["scale"], dtype=np.float32),
        )
