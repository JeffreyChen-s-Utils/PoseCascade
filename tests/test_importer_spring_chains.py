"""Tests for the glTF importer's spring-chain auto-detection.

Covers two layers:
  * Unit-level: the ``_attach_spring_chains`` helper attaches components to
    chain anchors, given a synthetic Skin.
  * Integration: loading the bundled ``examples/assets/character.glb`` (which
    has ``hair_*`` and ``orn_*`` chains we rigged in Blender) populates anchor
    nodes with one :class:`SpringChainComponent` per detected chain.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# `gltf` is the importer package — added to sys.path by tests/conftest.py.
from gltf.importer import _attach_spring_chains

from posecascade.assets.types import Skin
from posecascade.scene.component import SpringChainComponent
from posecascade.scene.node import Node
from posecascade.scene.transform import Transform
from posecascade.utils.math3d import vec3


def _make_skin_with_chains() -> Skin:
    anchor = Node(name="head_anchor")
    chain_c = []
    parent = anchor
    for i in range(4):
        node = Node(name=f"hair_C_{i}", transform=Transform(translation=vec3(0.0, -0.05, 0.0)))
        parent.add_child(node)
        chain_c.append(node)
        parent = node

    chain_l = []
    parent = anchor
    for i in range(4):
        node = Node(name=f"hair_L_{i}", transform=Transform(translation=vec3(-0.02, -0.05, 0.0)))
        parent.add_child(node)
        chain_l.append(node)
        parent = node

    orn = []
    parent = anchor
    for i in range(2):
        node = Node(name=f"orn_{i}", transform=Transform(translation=vec3(0.01, -0.02, 0.0)))
        parent.add_child(node)
        orn.append(node)
        parent = node

    joints = (anchor, *chain_c, *chain_l, *orn)
    return Skin(
        name="HairArmature",
        joints=joints,
        inverse_bind_matrices=np.tile(np.eye(4, dtype=np.float32), (len(joints), 1, 1)),
    )


def test_attach_spring_chains_creates_one_component_per_chain() -> None:
    skin = _make_skin_with_chains()
    _attach_spring_chains((skin,))
    anchor = skin.joints[0]
    components = [c for c in anchor.components if isinstance(c, SpringChainComponent)]
    names = sorted(c.chain_name for c in components)
    assert names == ["hair_C", "hair_L", "orn"]


def test_attach_spring_chains_uses_profile_defaults() -> None:
    skin = _make_skin_with_chains()
    _attach_spring_chains((skin,))
    anchor = skin.joints[0]
    components = {c.chain_name: c for c in anchor.components if isinstance(c, SpringChainComponent)}
    # Hair preset is softer than ornament preset.
    assert components["hair_C"].stiffness < components["orn"].stiffness
    # Hair joints tuple covers all four bones in chain.
    assert len(components["hair_C"].joints) == 4
    assert len(components["orn"].joints) == 2


def test_attach_spring_chains_is_idempotent() -> None:
    """Re-running detection on the same skin must not duplicate components."""
    skin = _make_skin_with_chains()
    _attach_spring_chains((skin,))
    _attach_spring_chains((skin,))
    anchor = skin.joints[0]
    components = [c for c in anchor.components if isinstance(c, SpringChainComponent)]
    names = sorted(c.chain_name for c in components)
    assert names == ["hair_C", "hair_L", "orn"]


def test_attach_spring_chains_skips_skin_without_chains() -> None:
    # Anchor + a single non-pattern bone — nothing to detect.
    anchor = Node(name="root")
    other = Node(name="head_anchor")
    anchor.add_child(other)
    skin = Skin(
        name="empty",
        joints=(anchor, other),
        inverse_bind_matrices=np.eye(4, dtype=np.float32).reshape(1, 4, 4).repeat(2, axis=0),
    )
    _attach_spring_chains((skin,))
    assert all(not isinstance(c, SpringChainComponent) for c in anchor.components)
    assert all(not isinstance(c, SpringChainComponent) for c in other.components)


# --- Integration with the bundled character.glb ----------------------------

_CHARACTER_GLB = Path(__file__).resolve().parent.parent / "examples" / "assets" / "character.glb"


@pytest.mark.skipif(not _CHARACTER_GLB.exists(), reason="examples/assets/character.glb missing")
def test_character_glb_imports_with_hair_and_ornament_chains() -> None:
    """End-to-end: loading the bundled character.glb must auto-attach 6 chains."""
    from gltf.importer import GltfImporter  # noqa: PLC0415 — local import inside guard

    imported = GltfImporter().load(_CHARACTER_GLB)
    expected = {"hair_LL", "hair_L", "hair_C", "hair_R", "hair_RR", "orn"}
    found_names: set[str] = set()
    for node in imported.scene.root.traverse():
        for component in node.components:
            if isinstance(component, SpringChainComponent):
                found_names.add(component.chain_name)
    # A chain anchor outside the active scene tree (Blender exports skin joints
    # as a flat extra group) — also walk every skin's joints to be safe.
    for skin in imported.scene.root.traverse():
        for component in skin.components:
            if isinstance(component, SpringChainComponent):
                found_names.add(component.chain_name)
    for skin in imported.skins:
        for joint in skin.joints:
            assert isinstance(joint, Node)
            for component in joint.components:
                if isinstance(component, SpringChainComponent):
                    found_names.add(component.chain_name)
    assert expected.issubset(found_names), f"missing chains: {expected - found_names}"
