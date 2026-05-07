"""Qt smoke tests for the effect chain dock."""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QFormLayout

from posecascade.render.effects.chain import EffectChain
from posecascade.render.effects.descriptor import (
    EffectDescriptor,
    EffectUniform,
    EffectUniformKind,
)
from posecascade.ui.effect_chain_dock import EffectChainDock


def _toy_descriptor(name: str) -> EffectDescriptor:
    return EffectDescriptor(
        name=name,
        fragment_shader=f"{name}.frag",
        uniforms=(
            EffectUniform(
                name="amount",
                kind=EffectUniformKind.SCALAR,
                default=0.5,
                minimum=0.0,
                maximum=1.0,
                step=0.01,
            ),
            EffectUniform(
                name="enable_flag",
                kind=EffectUniformKind.BOOL,
                default=True,
            ),
        ),
    )


def _seeded_chain() -> EffectChain:
    chain = EffectChain()
    chain.append(_toy_descriptor("tint"))
    chain.append(_toy_descriptor("vignette"))
    return chain


# ----- dock construction -------------------------------------------
def test_dock_lists_each_entry(qapp: object) -> None:
    chain = _seeded_chain()
    dock = EffectChainDock(chain=chain)
    assert dock._chain_list.count() == 2     # noqa: SLF001 — test seam


def test_dock_emits_chain_changed_when_toggled(qapp: object) -> None:
    chain = _seeded_chain()
    dock = EffectChainDock(chain=chain)
    received: list[None] = []
    dock.chain_changed.connect(lambda: received.append(None))
    dock._rows[0].enabled.setChecked(False)     # noqa: SLF001
    assert received
    assert chain.entries[0].enabled is False


# ----- reorder ------------------------------------------------------
def test_dock_move_up_swaps_entries(qapp: object) -> None:
    chain = _seeded_chain()
    dock = EffectChainDock(chain=chain)
    dock._chain_list.setCurrentRow(1)         # noqa: SLF001
    dock.move_selected_up()
    assert [entry.descriptor.name for entry in chain.entries] == ["vignette", "tint"]


def test_dock_move_down_at_bottom_is_noop(qapp: object) -> None:
    chain = _seeded_chain()
    dock = EffectChainDock(chain=chain)
    dock._chain_list.setCurrentRow(1)         # noqa: SLF001
    dock.move_selected_down()                 # already at the bottom
    assert [entry.descriptor.name for entry in chain.entries] == ["tint", "vignette"]


def test_dock_remove_drops_selected_entry(qapp: object) -> None:
    chain = _seeded_chain()
    dock = EffectChainDock(chain=chain)
    dock._chain_list.setCurrentRow(0)         # noqa: SLF001
    dock.remove_selected()
    assert len(chain) == 1
    assert chain.entries[0].descriptor.name == "vignette"


# ----- uniform editing ---------------------------------------------
def test_dock_uniform_edit_writes_override(qapp: object) -> None:
    chain = _seeded_chain()
    dock = EffectChainDock(chain=chain)
    dock._chain_list.setCurrentRow(0)         # noqa: SLF001
    # The uniforms form is rebuilt on selection. Find the spin box and
    # nudge it.
    layout = dock._uniforms_layout            # noqa: SLF001
    spin = layout.itemAt(0, QFormLayout.ItemRole.FieldRole).widget()
    spin.setValue(0.75)
    assert chain.entries[0].effective_value("amount") == pytest.approx(0.75)


def test_dock_bool_uniform_round_trip(qapp: object) -> None:
    chain = _seeded_chain()
    dock = EffectChainDock(chain=chain)
    dock._chain_list.setCurrentRow(0)         # noqa: SLF001
    layout = dock._uniforms_layout            # noqa: SLF001
    bool_widget = layout.itemAt(1, QFormLayout.ItemRole.FieldRole).widget()
    bool_widget.setChecked(False)
    assert chain.entries[0].effective_value("enable_flag") is False


def test_dock_refresh_preserves_selection(qapp: object) -> None:
    chain = _seeded_chain()
    dock = EffectChainDock(chain=chain)
    dock._chain_list.setCurrentRow(1)         # noqa: SLF001
    dock.refresh()
    assert dock._chain_list.currentRow() == 1     # noqa: SLF001


# Keep ``pytest`` reachable for IDE cross-reference.
__all__ = ["pytest"]
