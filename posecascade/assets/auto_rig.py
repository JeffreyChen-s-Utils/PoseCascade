"""Post-import auto-rig hooks shared across every format.

A few engine features key off bone-name conventions rather than format-
specific metadata — spring physics chains in particular. Historically
the glTF importer was the only one to apply them (its ``_attach_spring_
chains`` ran inside ``load()``), which left PMD / PMX / FBX imports
without auto-rigged hair, ribbons, skirt swing, etc.

This module centralises that logic. :func:`apply_post_import_rig`
runs after :class:`ImporterManager.load` regardless of the source
format, so every importer benefits without duplicating per-plugin
code. The operation is **idempotent** — re-running it on a scene that
already carries the relevant components is a no-op.
"""
from __future__ import annotations

from posecascade.animation.bone_aliasing import detect_humanoid_aliases
from posecascade.animation.spring import (
    detect_chains,
    resolve_chain_params_or_none,
)
from posecascade.assets.types import ImportedScene
from posecascade.scene.component import SpringChainComponent
from posecascade.scene.node import Node


def apply_post_import_rig(imported: ImportedScene) -> None:
    """Run every format-agnostic auto-rig pass on ``imported``.

    Currently:

    - **Spring chains**: walks every skin's joints, detects chains by
      name pattern (``hair_C_0..3`` or MMD-style ``前髪01..08``), tags
      the chain's anchor with a :class:`SpringChainComponent` carrying
      tuned stiffness / damping / inertia for the matching profile.
    - **Bone aliases**: pattern-matches the rig's joint names to
      canonical humanoid keys (``head``, ``upper_arm_L``, ...) and
      stores them on ``scene.bone_aliases`` so user scripts targeting
      anatomical names work across rigs without per-asset config.

    The glTF importer historically did this inside its own ``load()``
    so existing tests still see the attachment from there; this hook
    is idempotent (guarded by :func:`_has_chain_component`) so the
    duplicate call is harmless.
    """
    _attach_spring_chains(imported)
    _populate_bone_aliases(imported)


def _populate_bone_aliases(imported: ImportedScene) -> None:
    """Detect canonical humanoid bones and stash them on ``scene.bone_aliases``.

    Idempotent — re-running on an already-aliased scene replaces the
    map with the fresh detection result (callers that mutated bones
    or replaced skins between calls should see the update). Skinless
    scenes (props, environments) get an empty map and continue to work
    via the regular literal-name search.
    """
    all_joints: list[Node] = []
    for skin in imported.skins:
        all_joints.extend(skin.joints)
    imported.scene.bone_aliases = detect_humanoid_aliases(all_joints)


def _attach_spring_chains(imported: ImportedScene) -> None:
    """Detect + tag spring chains on every skin in ``imported``.

    Skips chains whose prefix has no entry in
    :data:`DEFAULT_CHAIN_PROFILES` — the naming-pattern detector
    catches every ``<prefix><digits>`` bone group, but most of those
    (finger joints, IK twist bones, append bones) are not physics
    chains. Only known-physics prefixes (``hair``, ``前髪``, ``リボン``,
    ``スカート``, …) actually get rigged.
    """
    for skin in imported.skins:
        for chain in detect_chains(skin.joints):
            if _has_chain_component(chain.anchor, chain.name):
                continue
            params = resolve_chain_params_or_none(chain.name)
            if params is None:
                continue
            chain.anchor.add_component(
                SpringChainComponent(
                    chain_name=chain.name,
                    joints=chain.joints,
                    stiffness=params.stiffness,
                    damping=params.damping,
                    inertia=params.inertia,
                ),
            )


def _has_chain_component(anchor: Node, chain_name: str) -> bool:
    return any(
        isinstance(c, SpringChainComponent) and c.chain_name == chain_name
        for c in anchor.components
    )
