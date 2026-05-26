"""Tests for the humanoid auto-collider emitter.

Pure-logic tests: build a fake alias map of ``{canonical_name → Node}``
where each ``Node`` is a stub with a ``name`` attribute, run the
emitter through a stub ``ColliderSpec`` factory, and assert the
emitted list matches the expected hip-sphere + thigh+shin capsule
pattern. Engine-layer behaviour, no Qt / GL / cloth solver involved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from posecascade.animation.auto_colliders import emit_humanoid_body_colliders


@dataclass
class _FakeNode:
    name: str


@dataclass(frozen=True)
class _FakeColliderSpec:
    kind: str
    follow_bone: str
    radius: float
    end_bone: str | None = None
    skin_offset: float = 0.005


def _alias_map(**bones: str) -> dict[str, Any]:
    """Build ``{canonical → _FakeNode(real_name)}`` from kwargs."""
    return {k: _FakeNode(name=v) for k, v in bones.items()}


def test_emits_full_humanoid_set_when_all_aliases_present() -> None:
    aliases = _alias_map(
        hip="Root_M_01",
        upper_leg_L="Hip_L_02", lower_leg_L="Knee_L_04", foot_L="Ankle_L_05",
        upper_leg_R="Hip_R_07", lower_leg_R="Knee_R_09", foot_R="Ankle_R_010",
    )
    specs = emit_humanoid_body_colliders(aliases, spec_factory=_FakeColliderSpec)
    assert len(specs) == 5
    # 1 sphere (hip) + 4 capsules (L thigh, R thigh, L shin, R shin)
    spheres = [s for s in specs if s.kind == "sphere"]
    capsules = [s for s in specs if s.kind == "capsule"]
    assert len(spheres) == 1
    assert len(capsules) == 4
    assert spheres[0].follow_bone == "Root_M_01"


def test_capsule_uses_real_bone_names_from_aliases() -> None:
    aliases = _alias_map(
        upper_leg_L="J_Bip_L_UpperLeg",
        lower_leg_L="J_Bip_L_LowerLeg",
    )
    specs = emit_humanoid_body_colliders(aliases, spec_factory=_FakeColliderSpec)
    thigh = next(s for s in specs if s.kind == "capsule")
    assert thigh.follow_bone == "J_Bip_L_UpperLeg"
    assert thigh.end_bone == "J_Bip_L_LowerLeg"


def test_missing_end_alias_drops_only_that_capsule() -> None:
    """Foot-less rig still gets the hip sphere + thigh capsules."""
    aliases = _alias_map(
        hip="Root",
        upper_leg_L="LegL", lower_leg_L="ShinL",  # foot_L missing
        upper_leg_R="LegR", lower_leg_R="ShinR",  # foot_R missing
    )
    specs = emit_humanoid_body_colliders(aliases, spec_factory=_FakeColliderSpec)
    kinds = sorted(s.kind for s in specs)
    # hip sphere + 2 thigh capsules; no shin capsules because foot bones absent
    assert kinds == ["capsule", "capsule", "sphere"]


def test_empty_alias_map_emits_nothing() -> None:
    specs = emit_humanoid_body_colliders({}, spec_factory=_FakeColliderSpec)
    assert specs == []


def test_only_hip_emits_only_the_sphere() -> None:
    aliases = _alias_map(hip="Root_M_01")
    specs = emit_humanoid_body_colliders(aliases, spec_factory=_FakeColliderSpec)
    assert len(specs) == 1
    assert specs[0].kind == "sphere"


def test_skin_offset_propagates() -> None:
    aliases = _alias_map(hip="Root", upper_leg_L="L", lower_leg_L="K")
    specs = emit_humanoid_body_colliders(
        aliases, spec_factory=_FakeColliderSpec, skin_offset=0.025,
    )
    assert all(s.skin_offset == 0.025 for s in specs)


def test_radii_match_documented_defaults() -> None:
    """The hip sphere is 13 cm, thighs 11 cm, shins 5 cm — pelvis +
    thighs are inflated past anatomical so a draping skirt has clear
    air to settle against even at peak step-cycle leg-lift."""
    aliases = _alias_map(
        hip="H",
        upper_leg_L="UL", lower_leg_L="LL", foot_L="FL",
        upper_leg_R="UR", lower_leg_R="LR", foot_R="FR",
    )
    specs = emit_humanoid_body_colliders(aliases, spec_factory=_FakeColliderSpec)
    by_kind: dict[tuple[str, str], float] = {
        (s.kind, s.follow_bone): s.radius for s in specs
    }
    assert by_kind[("sphere", "H")] == pytest.approx(0.16)
    assert by_kind[("capsule", "UL")] == pytest.approx(0.14)
    assert by_kind[("capsule", "LL")] == pytest.approx(0.08)
