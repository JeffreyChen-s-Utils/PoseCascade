"""Resolve a ``{morph_name → weight}`` map into per-leaf-morph weights.

Group morphs expand recursively (the parent's weight multiplies the
child's stored weight). Flip morphs forward the parent's weight to a
*single* selected child — the one whose ``parent_weight × child_weight``
product is highest among the entries — and zero out the rest.

The accumulator does NOT apply the resulting weights to GPU state; it
only collapses the tree into the leaf form a downstream applier can
process. Keeping it pure makes the recursion easy to test against
contrived hierarchies that would be cumbersome to express with full
GPU-side glue.
"""
from __future__ import annotations

from dataclasses import dataclass

from posecascade.animation.morph import (
    FlipMorph,
    GroupMorph,
    GroupMorphChild,
    MorphAsset,
)

_MIN_WEIGHT = 1.0e-6


@dataclass(frozen=True)
class LeafWeights:
    """Per-leaf morph weights after group / flip expansion.

    Keys are morph indices into :attr:`MorphAsset.by_index` referring to
    leaf morphs only (Vertex / Bone / UV / Material / Impulse). Group and
    Flip morphs do not appear here.
    """

    weights: dict[int, float]


def accumulate_weights(
    asset: MorphAsset,
    inputs: dict[str, float],
) -> LeafWeights:
    """Resolve ``inputs`` (name → weight) into leaf indices via ``asset``.

    Inputs whose names don't exist in ``asset`` are silently dropped — VMD
    motions are routinely re-targeted onto models with a different morph
    set and the per-frame application path is hot enough that we don't
    want to log every miss.
    """
    leaf: dict[int, float] = {}
    for name, weight in inputs.items():
        if abs(weight) < _MIN_WEIGHT:
            continue
        index = asset.by_name.get(name)
        if index is None:
            continue
        _visit(asset, index, float(weight), leaf, set())
    return LeafWeights(weights=leaf)


def accumulate_indexed_weights(
    asset: MorphAsset,
    inputs: dict[int, float],
) -> LeafWeights:
    """Index-keyed variant of :func:`accumulate_weights`.

    Used by group expansion when child references already live in the
    same address space as the recursion's accumulator.
    """
    leaf: dict[int, float] = {}
    for index, weight in inputs.items():
        if abs(weight) < _MIN_WEIGHT:
            continue
        _visit(asset, index, float(weight), leaf, set())
    return LeafWeights(weights=leaf)


def _visit(
    asset: MorphAsset,
    index: int,
    weight: float,
    leaf: dict[int, float],
    seen: set[int],
) -> None:
    """Walk one node of the morph tree, dispatching by type.

    Cycle protection: ``seen`` records every group / flip ancestor on the
    current recursion path. Encountering a cycle (a malformed model whose
    group references itself transitively) drops that branch silently.
    """
    if index < 0 or index >= len(asset.by_index):
        return
    if index in seen:
        return
    morph = asset.by_index[index]
    if isinstance(morph, GroupMorph):
        seen = seen | {index}
        for child in morph.children:
            child_weight = weight * float(child.weight)
            if abs(child_weight) < _MIN_WEIGHT:
                continue
            _visit(asset, child.morph_index, child_weight, leaf, seen)
        return
    if isinstance(morph, FlipMorph):
        chosen = _select_flip_child(morph.children, weight)
        if chosen is None:
            return
        seen = seen | {index}
        _visit(asset, chosen.morph_index, weight, leaf, seen)
        return
    leaf[index] = leaf.get(index, 0.0) + weight


def _select_flip_child(
    children: tuple[GroupMorphChild, ...], parent_weight: float,
) -> GroupMorphChild | None:
    """Pick the child whose ``parent_weight × child_weight`` is the highest.

    MMD's flip morph is exclusive — only one child shows at a time. The
    "highest weighted child" interpretation matches the documented MMD
    UI behaviour (the slider scrolls through children based on weight).
    """
    if not children:
        return None
    best: GroupMorphChild | None = None
    best_score = -float("inf")
    for child in children:
        score = parent_weight * float(child.weight)
        if score > best_score:
            best_score = score
            best = child
    return best
