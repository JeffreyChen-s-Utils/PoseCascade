"""Tests for :class:`MorphApplier` — vertex / UV / bone / material paths."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pmx.importer import PmxImporter

from posecascade.animation.morph_accumulator import LeafWeights, accumulate_weights
from posecascade.animation.morph_apply import MorphApplier

_TINY_MORPHS_PMX = Path(__file__).resolve().parent / "fixtures" / "mmd" / "tiny_morphs.pmx"


def _imported():
    return PmxImporter().load(_TINY_MORPHS_PMX)


def test_vertex_morph_offsets_baseline() -> None:
    """Vertex morph "vert_pull" offsets vertex 0 by (0.5, 0, 0) at full weight."""
    scene = _imported()
    applier = MorphApplier(scene)
    leaf = accumulate_weights(scene.morphs, {"vert_pull": 1.0})
    applier.apply(leaf)
    base = scene.meshes[0].positions
    np.testing.assert_allclose(
        applier.current_positions[0], base[0] + np.array([0.5, 0.0, 0.0]),
    )
    np.testing.assert_allclose(
        applier.current_positions[1], base[1] + np.array([0.0, 0.5, 0.0]),
    )
    # Untouched vertices stay at base.
    np.testing.assert_allclose(applier.current_positions[2], base[2])


def test_vertex_morph_at_zero_weight_resets_to_base() -> None:
    scene = _imported()
    applier = MorphApplier(scene)
    base_copy = scene.meshes[0].positions.copy()
    applier.apply(accumulate_weights(scene.morphs, {"vert_pull": 1.0}))
    assert not np.allclose(applier.current_positions, base_copy)
    applier.apply(LeafWeights(weights={}))
    np.testing.assert_allclose(applier.current_positions, base_copy)


def test_bone_morph_translation_and_rotation_returned() -> None:
    scene = _imported()
    applier = MorphApplier(scene)
    snapshot = applier.apply(accumulate_weights(scene.morphs, {"bone_tilt": 1.0}))
    # bone_tilt targets bone index 1 with translation (0.1, 0, 0) and a
    # 30°-around-Z quaternion at full weight.
    assert 1 in snapshot.bone_offsets
    translation, rotation = snapshot.bone_offsets[1]
    np.testing.assert_allclose(translation, [0.1, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(rotation, [0.0, 0.0, 0.2588190, 0.9659258], atol=1e-4)


def test_bone_morph_half_weight_scales_translation_and_slerps_rotation() -> None:
    scene = _imported()
    applier = MorphApplier(scene)
    snapshot = applier.apply(accumulate_weights(scene.morphs, {"bone_tilt": 0.5}))
    translation, rotation = snapshot.bone_offsets[1]
    np.testing.assert_allclose(translation, [0.05, 0.0, 0.0], atol=1e-5)
    # Halfway between identity and 30° → 15°
    expected_x = float(np.sin(np.deg2rad(7.5)))
    expected_w = float(np.cos(np.deg2rad(7.5)))
    np.testing.assert_allclose(
        rotation, [0.0, 0.0, expected_x, expected_w], atol=1e-4,
    )


def test_material_morph_multiply_blends_diffuse() -> None:
    """``redder`` is a multiply-mode morph with diffuse factor (1.5, 0.5, 0.5, 1).

    At ``weight = 1`` the resulting diffuse should equal ``base * factor``.
    """
    scene = _imported()
    applier = MorphApplier(scene)
    snapshot = applier.apply(accumulate_weights(scene.morphs, {"redder": 1.0}))
    base = scene.meshes[0].mmd_material.diffuse
    expected = (base[0] * 1.5, base[1] * 0.5, base[2] * 0.5, base[3] * 1.0)
    np.testing.assert_allclose(snapshot.material_overrides[0].diffuse, expected, atol=1e-5)


def test_material_morph_half_weight_lerps_factor_toward_one() -> None:
    scene = _imported()
    applier = MorphApplier(scene)
    snapshot = applier.apply(accumulate_weights(scene.morphs, {"redder": 0.5}))
    base = scene.meshes[0].mmd_material.diffuse
    # Multiplicative-morph factor lerps from 1 toward target by weight.
    factor = (1.0 + 0.5 * 0.5, 1.0 + 0.5 * (-0.5), 1.0 + 0.5 * (-0.5), 1.0)
    expected = tuple(b * f for b, f in zip(base, factor, strict=True))
    np.testing.assert_allclose(snapshot.material_overrides[0].diffuse, expected, atol=1e-5)


def test_group_morph_drives_vertex_and_bone_at_once() -> None:
    """The ``all`` group has child indices 0 (vertex) and 1 (bone) at weight 1."""
    scene = _imported()
    applier = MorphApplier(scene)
    snapshot = applier.apply(accumulate_weights(scene.morphs, {"all": 1.0}))
    # Vertex morph took effect.
    np.testing.assert_allclose(
        applier.current_positions[0],
        scene.meshes[0].positions[0] + np.array([0.5, 0.0, 0.0]),
    )
    # Bone morph delta surfaces in the snapshot.
    assert 1 in snapshot.bone_offsets


def test_dirty_flags_clear_after_mark_uploaded() -> None:
    scene = _imported()
    applier = MorphApplier(scene)
    applier.apply(accumulate_weights(scene.morphs, {"vert_pull": 1.0}))
    assert applier.positions_dirty
    applier.mark_uploaded()
    assert not applier.positions_dirty


def test_uv_morph_offsets_primary_channel_only(tmp_path: Path) -> None:
    """Constructing a PMX with a UV morph (channel 0) should affect texcoords.

    UV1..4 morphs are parsed but currently skipped by the applier — the
    test asserts non-zero only for UV0 by setting weight to 1 and reading
    the applier's ``current_texcoords``.
    """
    from dataclasses import replace  # noqa: PLC0415

    from tests.fixtures.mmd.build import (  # noqa: PLC0415 — local helper import
        FixtureMorph,
        FixtureUvMorphOffset,
        build_pmx,
        tiny_cube_with_morphs_spec,
    )
    base = tiny_cube_with_morphs_spec()
    extra = FixtureMorph(
        name="uv_shift",
        morph_type=3,                                       # UV (channel 0)
        uv_offsets=(
            FixtureUvMorphOffset(vertex_index=0, offset=(0.25, 0.0, 0.0, 0.0)),
        ),
    )
    spec = replace(base, morphs=(*base.morphs, extra))
    path = tmp_path / "with_uv.pmx"
    path.write_bytes(build_pmx(spec))
    scene = PmxImporter().load(path)
    applier = MorphApplier(scene)
    applier.apply(accumulate_weights(scene.morphs, {"uv_shift": 1.0}))
    assert applier.current_texcoords[0, 0] == pytest.approx(0.25, abs=1e-5)


def test_player_streams_morph_state_to_renderer_double() -> None:
    """When the player has a renderer attached, ``apply`` should call the
    renderer's stream + override hooks. Use a duck-typed double — no GL
    context needed."""
    from vmd.importer import VmdImporter  # noqa: PLC0415

    from posecascade.animation.player import VmdAnimationPlayer  # noqa: PLC0415

    class RendererDouble:
        def __init__(self):
            self.streamed_calls = 0
            self.last_overrides = None
        def stream_morphed_buffers(self, positions, texcoords):
            self.streamed_calls += 1
        def set_material_overrides(self, overrides):
            self.last_overrides = overrides

    scene = _imported()
    motion = VmdImporter().load(
        Path(__file__).resolve().parent / "fixtures" / "mmd" / "morphs.vmd"
    )
    rd = RendererDouble()
    player = VmdAnimationPlayer.for_imported_scene(motion, scene, renderer=rd)
    player.apply(0.0)
    assert rd.streamed_calls == 1
    assert isinstance(rd.last_overrides, dict)
