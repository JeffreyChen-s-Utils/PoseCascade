"""Engine-side driver that builds spring chains from scene components and steps each frame.

The :class:`PhysicsHost` is the bridge between scene data (an
:class:`~posecascade.scene.component.SpringChainComponent` attached by the
importer) and the simulation core (:class:`~posecascade.animation.spring.SpringSimulator`).
The bootstrap layer registers each loaded scene with the host, and the main
window tick calls :meth:`PhysicsHost.tick` once per frame — before the script
host runs, so user scripts read the latest physics-driven joint poses.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from posecascade.animation.cloth import CapsuleCollider, SphereCollider
from posecascade.animation.spring import (
    ExternalForce,
    Gravity,
    SpringChain,
    SpringParams,
    SpringSimulator,
)
from posecascade.assets.types import ImportedScene
from posecascade.scene.component import SpringChainComponent
from posecascade.scene.node import Node
from posecascade.scene.scene import Scene
from posecascade.utils.logging import get_logger
from posecascade.utils.math3d import vec3

_log = get_logger(__name__)
# Default gentle world-space gravity — enough to give hair "weight" without
# forcing the user to tune anything for a first run.
_DEFAULT_GRAVITY_FORCE = (0.0, -2.5, 0.0)


@dataclass
class PhysicsHost:
    """Owns a single :class:`SpringSimulator` and auto-registers chains from scenes."""

    _sim: SpringSimulator = field(default_factory=SpringSimulator)
    _registered_anchors: set[int] = field(default_factory=set)
    _gravity_installed: bool = False

    @property
    def simulator(self) -> SpringSimulator:
        return self._sim

    def install_default_forces(self) -> None:
        """Install the default gravity force (idempotent — safe to call repeatedly)."""
        if self._gravity_installed:
            return
        self._sim.add_force(Gravity(force=vec3(*_DEFAULT_GRAVITY_FORCE)))
        self._gravity_installed = True

    def register_imported_scene(self, imported: ImportedScene) -> None:
        """Scan ``imported.scene`` and ``imported.skins`` for spring chains and rig them."""
        if imported.scene is not None:
            self.register_scene(imported.scene)
        for skin in imported.skins:
            self.register_joints(skin.joints)

    def register_scene(self, scene: Scene) -> None:
        """Walk every node in ``scene`` and rig chains tagged via :class:`SpringChainComponent`."""
        for node in scene.root.traverse():
            self._register_node(node)

    def register_joints(self, joints: Sequence[object]) -> None:
        """Register chains found on any of ``joints`` — useful for glTF skins whose joints
        live outside the active scene tree."""
        for joint in joints:
            if isinstance(joint, Node):
                self._register_node(joint)

    def tick(self, dt: float) -> None:
        """Advance the simulation by ``dt`` seconds."""
        self._sim.step(dt)

    def add_force(self, force: ExternalForce) -> None:
        """Add a global external force (gravity / wind / point force) applied to every chain."""
        self._sim.add_force(force)

    def add_collider(self, collider: SphereCollider | CapsuleCollider) -> None:
        """Register one body collider so hair / ribbon chains push out of it.

        See :attr:`SpringSimulator.colliders`. Prefer :meth:`share_colliders_with`
        when the cloth host already manages a collider list — that way a
        single ``bone-follow`` driver update is observed by both physics
        systems and the two stay in lock-step automatically.
        """
        self._sim.add_collider(collider)

    def share_colliders_with(self, cloth_host: object) -> None:
        """Adopt ``cloth_host``'s collider list by reference (zero-copy).

        Replaces the simulator's own ``colliders`` field with the same
        Python list the cloth host mutates each frame. Future ``add_collider``
        calls on either side end up in the shared list, so hair-vs-body
        and cloth-vs-body see identical capsules without a synchronization
        pass. Idempotent — calling twice with the same host is a no-op
        for the second call.
        """
        # The cloth host's public ``colliders`` method returns a tuple snapshot;
        # we want the live list it mutates. That list lives on the underlying
        # solver — go through ``_solver.colliders`` so future ``add_collider``
        # calls on either host land in the same Python list.
        solver = getattr(cloth_host, "_solver", None)
        host_colliders = getattr(solver, "colliders", None) if solver else None
        if host_colliders is None:
            return
        if self._sim.colliders is host_colliders:
            return
        self._sim.colliders = host_colliders

    def find_chain(self, name: str) -> SpringChain | None:
        """Look up a chain by its name (e.g. ``"hair_C"``); ``None`` if not registered."""
        return self._sim.find_chain(name)

    def chains(self) -> tuple[SpringChain, ...]:
        return tuple(self._sim.chains)

    def reset(self) -> None:
        """Rebuild from scratch: drop all chains/forces and forget registered anchors."""
        self._sim = SpringSimulator()
        self._registered_anchors.clear()
        self._gravity_installed = False

    def remove_chains_for_subtree(self, root_node: Node) -> int:
        """Drop every chain whose anchor or any joint sits under ``root_node``.

        Used by the editor when a subtree is deleted from the scene — without
        this, the simulator keeps stepping chains pointing at orphaned nodes
        and bone matrices visibly snap to garbage. Returns the count removed
        so callers can log meaningful feedback.
        """
        subtree_ids = {id(n) for n in root_node.traverse()}
        removed = 0
        kept: list[SpringChain] = []
        for chain in self._sim.chains:
            in_subtree = id(chain.anchor) in subtree_ids or any(
                id(joint.node) in subtree_ids for joint in chain.joints
            )
            if in_subtree:
                # Remove the registered_anchor key so the same anchor can be
                # re-added later if the user re-imports the same scene.
                self._registered_anchors.discard(id(chain.anchor) ^ hash(chain.name))
                removed += 1
            else:
                kept.append(chain)
        self._sim.chains = kept
        return removed

    def _register_node(self, node: Node) -> None:
        for component in node.components:
            if isinstance(component, SpringChainComponent):
                self._build_chain(node, component)

    def _build_chain(self, anchor: Node, component: SpringChainComponent) -> None:
        anchor_key = id(anchor) ^ hash(component.chain_name)
        if anchor_key in self._registered_anchors:
            return
        if not component.joints:
            _log.warning("chain %r has no joints — skipping", component.chain_name)
            return
        params = SpringParams(
            stiffness=component.stiffness,
            damping=component.damping,
            inertia=component.inertia,
        )
        chain = SpringChain.from_node_chain(
            component.chain_name,
            anchor,
            list(component.joints),
            params=params,
        )
        self._sim.add_chain(chain)
        self._registered_anchors.add(anchor_key)
