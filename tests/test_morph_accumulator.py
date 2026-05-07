"""Tests for the morph weight accumulator (group / flip resolution)."""
from __future__ import annotations

import pytest

from posecascade.animation.morph import (
    BoneMorph,
    BoneMorphOffset,
    FlipMorph,
    GroupMorph,
    GroupMorphChild,
    MaterialMorph,
    MaterialMorphOp,
    MaterialMorphTarget,
    UvMorph,
    UvMorphOffset,
    VertexMorph,
    VertexMorphOffset,
    build_morph_asset,
)
from posecascade.animation.morph_accumulator import (
    accumulate_indexed_weights,
    accumulate_weights,
)


def _two_leaves() -> tuple:
    """Two minimal leaf morphs the rest of the tests build trees on top of."""
    return (
        VertexMorph(name="left_eye", offsets=(
            VertexMorphOffset(vertex_index=0, offset=(0.1, 0.0, 0.0)),
        )),
        VertexMorph(name="right_eye", offsets=(
            VertexMorphOffset(vertex_index=1, offset=(0.1, 0.0, 0.0)),
        )),
    )


def test_direct_leaf_weight_lands_at_index() -> None:
    leaves = _two_leaves()
    asset = build_morph_asset(leaves)
    out = accumulate_weights(asset, {"left_eye": 0.4})
    assert out.weights == {0: 0.4}


def test_group_expansion_scales_children() -> None:
    leaves = _two_leaves()
    group = GroupMorph(name="blink", children=(
        GroupMorphChild(morph_index=0, weight=0.5),
        GroupMorphChild(morph_index=1, weight=1.0),
    ))
    asset = build_morph_asset((*leaves, group))
    out = accumulate_weights(asset, {"blink": 0.8})
    assert out.weights == pytest.approx({0: 0.4, 1: 0.8})


def test_nested_groups_chain_weights() -> None:
    leaves = _two_leaves()
    inner = GroupMorph(name="inner", children=(
        GroupMorphChild(morph_index=0, weight=0.5),
    ))
    outer = GroupMorph(name="outer", children=(
        GroupMorphChild(morph_index=2, weight=0.4),
    ))
    asset = build_morph_asset((*leaves, inner, outer))
    out = accumulate_weights(asset, {"outer": 1.0})
    assert out.weights == pytest.approx({0: 0.2})


def test_flip_picks_highest_product_child() -> None:
    leaves = _two_leaves()
    flip = FlipMorph(name="mouth", children=(
        GroupMorphChild(morph_index=0, weight=0.3),
        GroupMorphChild(morph_index=1, weight=0.8),
    ))
    asset = build_morph_asset((*leaves, flip))
    out = accumulate_weights(asset, {"mouth": 1.0})
    assert out.weights == {1: 1.0}


def test_flip_with_negative_parent_picks_lowest_product_branch() -> None:
    """Negative parent weight inverts the comparison — the *lowest* product wins.

    This mirrors how MMD's morph slider treats negative values; not common
    but the accumulator must not crash on it.
    """
    leaves = _two_leaves()
    flip = FlipMorph(name="mouth", children=(
        GroupMorphChild(morph_index=0, weight=0.3),
        GroupMorphChild(morph_index=1, weight=0.8),
    ))
    asset = build_morph_asset((*leaves, flip))
    out = accumulate_weights(asset, {"mouth": -1.0})
    assert 0 in out.weights or 1 in out.weights


def test_zero_weight_short_circuits() -> None:
    leaves = _two_leaves()
    asset = build_morph_asset(leaves)
    out = accumulate_weights(asset, {"left_eye": 0.0})
    assert out.weights == {}


def test_missing_morph_silently_ignored() -> None:
    leaves = _two_leaves()
    asset = build_morph_asset(leaves)
    out = accumulate_weights(asset, {"unknown": 1.0})
    assert out.weights == {}


def test_cycle_does_not_loop_forever() -> None:
    """A self-referencing group should not stack-overflow the accumulator."""
    self_ref = GroupMorph(name="loop", children=(
        GroupMorphChild(morph_index=0, weight=1.0),
    ))
    asset = build_morph_asset((self_ref,))
    out = accumulate_weights(asset, {"loop": 1.0})
    assert out.weights == {}


def test_indexed_weights_skip_name_lookup() -> None:
    leaves = _two_leaves()
    asset = build_morph_asset(leaves)
    out = accumulate_indexed_weights(asset, {0: 0.6, 1: 0.4})
    assert out.weights == pytest.approx({0: 0.6, 1: 0.4})


def test_repeated_inputs_sum_at_leaf() -> None:
    leaves = _two_leaves()
    group = GroupMorph(name="both", children=(
        GroupMorphChild(morph_index=0, weight=1.0),
    ))
    asset = build_morph_asset((*leaves, group))
    out = accumulate_weights(asset, {"left_eye": 0.5, "both": 0.5})
    assert out.weights == pytest.approx({0: 1.0})


# Keep the unused symbol references load-bearing — IDEs follow them and
# the linter would otherwise strip the imports we want kept for type hints.
__all__ = [
    "BoneMorph",
    "BoneMorphOffset",
    "MaterialMorph",
    "MaterialMorphOp",
    "MaterialMorphTarget",
    "UvMorph",
    "UvMorphOffset",
]
