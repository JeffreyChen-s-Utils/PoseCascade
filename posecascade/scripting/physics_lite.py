"""Curated physics surface exposed to user scripts.

Wraps :class:`~posecascade.animation.physics_host.PhysicsHost` in a minimal,
type-friendly API so a sandboxed script can tune chains and add forces without
touching engine internals or importing modules (the sandbox forbids ``import``).

The exposed names are: ``set_gravity``, ``add_wind``, ``add_point_force``,
``get_chain`` (returns a :class:`ChainHandle`), and ``chain_names``.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from posecascade.animation.cloth import (
    CapsuleCollider,
    ClothGravity,
    ClothPiece,
    ClothWind,
    SphereCollider,
)
from posecascade.animation.cloth_host import ClothHost
from posecascade.animation.physics_host import PhysicsHost
from posecascade.animation.spring import (
    Gravity,
    PointForce,
    SpringChain,
    Wind,
)
from posecascade.scene.node import Node
from posecascade.utils.math3d import Vec3

_VEC3_LENGTH = 3


def _to_vec3(value: object) -> Vec3:
    """Accept a 3-tuple, list, or numpy array and return a numpy float32 vec3."""
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape[0] != _VEC3_LENGTH:
        raise ValueError(f"expected a 3-component vector, got shape {arr.shape}")
    return arr.copy()


class ChainHandle:
    """Lightweight, mutable view over a single :class:`SpringChain` for user scripts."""

    def __init__(self, chain: SpringChain) -> None:
        self._chain = chain

    @property
    def name(self) -> str:
        return self._chain.name

    @property
    def enabled(self) -> bool:
        return self._chain.enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._chain.enabled = bool(value)

    @property
    def stiffness(self) -> float:
        return self._chain.stiffness

    @stiffness.setter
    def stiffness(self, value: float) -> None:
        self._chain.stiffness = float(value)

    @property
    def damping(self) -> float:
        return self._chain.damping

    @damping.setter
    def damping(self, value: float) -> None:
        self._chain.damping = float(value)

    def set_inertia(self, value: float) -> None:
        """Set the moment of inertia on every joint in this chain."""
        for joint in self._chain.joints:
            joint.inertia = float(value)

    def apply_impulse(self, axis: object, magnitude: float, joint_index: int = -1) -> None:
        """Add an angular-velocity impulse (rad/s) along ``axis`` to one joint.

        ``joint_index`` defaults to the tip joint. Use this for explosions, hits,
        or scripted reaction shakes — pure :meth:`add_force` only ramps via torque
        which feels mushy for instant kicks.
        """
        axis_vec = _to_vec3(axis)
        if joint_index < 0:
            joint_index += len(self._chain.joints)
        if joint_index < 0 or joint_index >= len(self._chain.joints):
            raise IndexError(f"joint_index {joint_index} out of range for chain {self.name!r}")
        joint = self._chain.joints[joint_index]
        joint.angular_velocity = (joint.angular_velocity + axis_vec * float(magnitude)).astype(
            np.float32
        )

    def reset(self) -> None:
        """Clear angular velocity and snap rotations back to the rest pose."""
        self._chain.reset()


class ClothPieceHandle:
    """Lightweight handle over a single :class:`ClothPiece` for user scripts."""

    def __init__(self, piece: ClothPiece) -> None:
        self._piece = piece

    @property
    def name(self) -> str:
        return self._piece.name

    @property
    def enabled(self) -> bool:
        return self._piece.enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._piece.enabled = bool(value)

    @property
    def structural_stiffness(self) -> float:
        return self._piece.params.structural_stiffness

    @structural_stiffness.setter
    def structural_stiffness(self, value: float) -> None:
        self._piece.params.structural_stiffness = float(value)

    @property
    def bend_stiffness(self) -> float:
        return self._piece.params.bend_stiffness

    @bend_stiffness.setter
    def bend_stiffness(self, value: float) -> None:
        self._piece.params.bend_stiffness = float(value)

    @property
    def linear_damping(self) -> float:
        return self._piece.params.linear_damping

    @linear_damping.setter
    def linear_damping(self, value: float) -> None:
        self._piece.params.linear_damping = float(value)

    def set_iterations(self, value: int) -> None:
        """Set PBD constraint solve iterations per substep (more = stiffer, slower)."""
        self._piece.params.iterations = int(value)

    @property
    def rest_pull(self) -> float:
        return self._piece.params.rest_pull

    @rest_pull.setter
    def rest_pull(self, value: float) -> None:
        self._piece.params.rest_pull = float(value)

    def reset(self) -> None:
        self._piece.reset()


class PhysicsLite:
    """User-script facade over :class:`PhysicsHost` (joint chains) + :class:`ClothHost` (cloth)."""

    def __init__(self, host: PhysicsHost, cloth_host: ClothHost | None = None) -> None:
        self._host = host
        self._cloth = cloth_host

    def set_gravity(self, force: object) -> None:
        """Replace (or install) the global gravity force vector."""
        new_force = _to_vec3(force)
        for existing in self._host.simulator.global_forces:
            if isinstance(existing, Gravity):
                existing.force = new_force
                return
        self._host.add_force(Gravity(force=new_force))

    def add_wind(
        self,
        direction: object,
        speed: float,
        turbulence_amplitude: float = 0.0,
        turbulence_frequency_hz: float = 2.0,
    ) -> Wind:
        """Add a directional wind. Returns the :class:`Wind` so the caller can mutate it later."""
        wind = Wind(
            direction=_to_vec3(direction),
            speed=float(speed),
            turbulence_amplitude=float(turbulence_amplitude),
            turbulence_frequency_hz=float(turbulence_frequency_hz),
        )
        self._host.add_force(wind)
        return wind

    def add_point_force(
        self,
        source: object,
        magnitude: float,
        falloff_distance: float,
    ) -> PointForce:
        """Add a radial force from ``source`` with linear falloff over ``falloff_distance``."""
        pf = PointForce(
            source=_to_vec3(source),
            magnitude=float(magnitude),
            falloff_distance=float(falloff_distance),
        )
        self._host.add_force(pf)
        return pf

    def get_chain(self, name: str) -> ChainHandle | None:
        """Look up a chain by name (e.g. ``"hair_C"``); returns ``None`` if not found."""
        chain = self._host.find_chain(name)
        if chain is None:
            return None
        return ChainHandle(chain)

    def chain_names(self) -> tuple[str, ...]:
        """Names of every registered chain (useful for discovery in scripts)."""
        return tuple(c.name for c in self._host.chains())

    def chains(self) -> Iterable[ChainHandle]:
        """Iterate :class:`ChainHandle` over every registered chain."""
        return tuple(ChainHandle(c) for c in self._host.chains())

    def reset_all(self) -> None:
        """Snap every chain back to its rest pose (zero velocity, identity local rotation)."""
        for chain in self._host.chains():
            chain.reset()

    # --- Cloth API ---------------------------------------------------------

    def add_cloth(
        self,
        node: object,
        *,
        mesh_index: int | None = None,
        cloth_name: str | None = None,
        anchor_axis: int = 1,
        anchor_fraction: float = 0.15,
        anchor_mode: str = "top_axis",
        simulate_top_below: float | None = None,
        structural_stiffness: float = 0.85,
        bend_stiffness: float = 0.10,
        linear_damping: float = 0.985,
        iterations: int = 8,
        rest_pull: float = 4.0,
        track_bone: object | None = None,
        skin_target: bool = False,
    ) -> ClothPieceHandle | None:
        """Mark ``node``'s mesh as a simulated cloth piece. Returns a handle (or ``None``).

        ``anchor_mode``: ``"top_axis"`` pins the top fraction of the whole mesh
        bbox; ``"per_island_top"`` pins the top fraction of each connected mesh
        island separately — use that for sleeve / decoration meshes whose
        upper-arm cap and lower flap are different topological islands.

        ``simulate_top_below`` (only with ``"per_island_top"``): islands whose
        max along ``anchor_axis`` exceeds this value are FULLY anchored — useful
        for filtering a multi-piece sleeve mesh down to just the lower flowing
        flaps while leaving the puffy upper part rigid against the body.

        ``track_bone`` (Node, optional): pin the cloth's anchor verts to this
        bone's world frame. Without it, when the character translates the
        anchored top of the cloth stays at its initial world position and
        the skirt visibly detaches from the moving body. Typically the hip
        / pelvis bone for a skirt, the upper-chest bone for a cape.

        ``skin_target`` (bool, default False): drive each cloth vert's rest
        position from the mesh's bone skinning weights every tick. The
        cloth's ``rest_pull`` term then pulls free verts toward this dynamic
        skinned target, so the cloth follows leg/arm swing while still
        allowing dynamic drape. Requires the cloth mesh to carry a skin
        (``SkinRefComponent`` + ``mesh.joints_0`` / ``weights_0``); silently
        no-ops on unskinned cloth.
        """
        self._require_cloth("add_cloth")
        if not isinstance(node, Node):
            raise TypeError(f"add_cloth expected a Node, got {type(node).__name__}")
        if track_bone is not None and not isinstance(track_bone, Node):
            raise TypeError(
                f"add_cloth track_bone expected a Node, got {type(track_bone).__name__}",
            )
        piece = self._cloth.add_cloth_for_node(
            node,
            mesh_index=mesh_index,
            cloth_name=cloth_name,
            anchor_axis=anchor_axis,
            anchor_fraction=anchor_fraction,
            anchor_mode=anchor_mode,
            simulate_top_below=simulate_top_below,
            structural_stiffness=structural_stiffness,
            bend_stiffness=bend_stiffness,
            linear_damping=linear_damping,
            iterations=iterations,
            rest_pull=rest_pull,
        )
        if piece is not None and track_bone is not None:
            self._cloth.register_anchor_follower(piece, track_bone)
        if piece is not None and skin_target:
            self._cloth.register_skin_target_follower(piece, node)
        return ClothPieceHandle(piece) if piece is not None else None

    def add_collision_deform(
        self,
        node: object,
        *,
        mesh_index: int | None = None,
    ) -> ClothPieceHandle | None:
        """Mark a skinned mesh for **post-skin collision push-out** every frame.

        Drives the mesh from its bone skinning each tick (rigidly — no
        cloth dynamics, no springs, no gravity) and then projects every
        vertex out of any registered colliders. Use this to stop dress,
        sleeve, or hair-fringe vertices from clipping into the torso /
        limbs during animation without re-rigging the asset — register
        the offending mesh here, add body-following capsule / sphere
        colliders via :meth:`add_capsule_collider` / :meth:`add_sphere_collider`,
        and the runtime pushes any vertex inside a collider back to its
        surface every frame.

        Internally this is implemented as a :class:`~posecascade.animation.cloth.ClothPiece`
        flagged ``passive_skin_deform=True`` — the solver skips Verlet
        integration and PBD constraints, so the per-frame cost on a
        30k-vert body mesh is ~1 ms versus ~6 ms for full cloth.

        Requires the mesh to carry a skin (``SkinRefComponent`` plus
        ``mesh.joints_0`` / ``weights_0``); silently no-ops on unskinned
        meshes.
        """
        self._require_cloth("add_collision_deform")
        if not isinstance(node, Node):
            raise TypeError(
                f"add_collision_deform expected a Node, got {type(node).__name__}",
            )
        piece = self._cloth.add_cloth_for_node(
            node,
            mesh_index=mesh_index,
            cloth_name=f"collision_deform_{node.name}",
            anchor_axis=1,
            anchor_fraction=0.0,
            anchor_mode="top_axis",
            structural_stiffness=0.0,
            bend_stiffness=0.0,
            linear_damping=1.0,
            iterations=1,
            rest_pull=0.0,
        )
        if piece is None:
            return None
        piece.params.passive_skin_deform = True
        self._cloth.register_skin_target_follower(piece, node)
        return ClothPieceHandle(piece)

    def add_sphere_collider(self, center: object, radius: float) -> SphereCollider:
        """Add a sphere collider that pushes cloth verts outside it (e.g. body proxy)."""
        self._require_cloth("add_sphere_collider")
        collider = SphereCollider(center=_to_vec3(center), radius=float(radius))
        self._cloth.add_collider(collider)
        return collider

    def add_capsule_collider(self, a: object, b: object, radius: float) -> CapsuleCollider:
        """Add a capsule collider between two world points (e.g. for legs / torso)."""
        self._require_cloth("add_capsule_collider")
        collider = CapsuleCollider(a=_to_vec3(a), b=_to_vec3(b), radius=float(radius))
        self._cloth.add_collider(collider)
        return collider

    def set_cloth_gravity(self, force: object) -> None:
        """Replace (or install) the cloth solver's gravity force."""
        self._require_cloth("set_cloth_gravity")
        new_force = _to_vec3(force)
        for existing in self._cloth.solver.forces:
            if isinstance(existing, ClothGravity):
                existing.acceleration = new_force
                return
        self._cloth.add_force(ClothGravity(acceleration=new_force))

    def add_cloth_wind(
        self,
        direction: object,
        speed: float,
        turbulence_amplitude: float = 0.0,
        turbulence_frequency_hz: float = 1.5,
    ) -> ClothWind:
        """Add a directional wind to the cloth solver. Returns the :class:`ClothWind` for tweaks."""
        self._require_cloth("add_cloth_wind")
        wind = ClothWind(
            direction=_to_vec3(direction),
            speed=float(speed),
            turbulence_amplitude=float(turbulence_amplitude),
            turbulence_frequency_hz=float(turbulence_frequency_hz),
        )
        self._cloth.add_force(wind)
        return wind

    def get_cloth(self, name: str) -> ClothPieceHandle | None:
        """Look up a cloth piece by name (e.g. the value passed to ``cloth_name=``)."""
        self._require_cloth("get_cloth")
        piece = self._cloth.find_piece(name)
        return ClothPieceHandle(piece) if piece is not None else None

    def cloth_names(self) -> tuple[str, ...]:
        if self._cloth is None:
            return ()
        return tuple(p.name for p in self._cloth.solver.pieces)

    def _require_cloth(self, method: str) -> None:
        if self._cloth is None:
            raise RuntimeError(f"physics_lite.{method} requires cloth_host (not configured)")
