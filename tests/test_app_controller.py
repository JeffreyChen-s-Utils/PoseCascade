"""Tests for :class:`AppController` orchestration."""
from __future__ import annotations

from pathlib import Path

import pytest
from pmx.importer import PmxImporter
from vmd.importer import VmdImporter

from posecascade.app.controller import AppController
from posecascade.render.effects.builtins import register_builtins
from posecascade.render.effects.chain import EffectChain, EffectLibrary
from posecascade.scene.model_slot import ModelSlot, SceneSlots
from posecascade.ui.export_dialog import ExportSpec, ExportTarget

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mmd"


def _controller(tmp_path: Path) -> AppController:
    return AppController(
        project_root=tmp_path,
        pmx_loader=PmxImporter().load,
        vmd_loader=VmdImporter().load,
    )


def _slot(name: str) -> ModelSlot:
    return ModelSlot(name=name, imported=PmxImporter().load(_FIXTURES / "tiny.pmx"))


def test_controller_registers_builtins_at_construction(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    assert "autoluminous" in controller.effect_library.names()
    assert "hgshadow" in controller.effect_library.names()


def test_add_slot_remembers_source_paths(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    pmx = tmp_path / "model.pmx"
    pmx.write_bytes((_FIXTURES / "tiny.pmx").read_bytes())
    slot = _slot("character")
    controller.add_slot(slot, model_path=pmx)
    assert controller._slot_source_paths["character"] == (pmx, None)    # noqa: SLF001


def test_clear_slots_drops_state(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    pmx = tmp_path / "model.pmx"
    pmx.write_bytes((_FIXTURES / "tiny.pmx").read_bytes())
    controller.add_slot(_slot("a"), model_path=pmx)
    controller.clear_slots()
    assert len(controller.slots) == 0


def test_on_frame_changed_runs_player_when_motion_present(tmp_path: Path) -> None:
    """A slot with a VMD motion should advance bone TRS on each
    on_frame_changed call. We don't care about the exact value, just
    that the player ran (i.e. some bone moved off rest)."""
    import numpy as np  # noqa: PLC0415 — late import
    pmx = tmp_path / "model.pmx"
    pmx.write_bytes((_FIXTURES / "tiny.pmx").read_bytes())
    vmd = tmp_path / "wave.vmd"
    vmd.write_bytes((_FIXTURES / "wave.vmd").read_bytes())
    slot = ModelSlot(
        name="alice",
        imported=PmxImporter().load(pmx),
        motion=VmdImporter().load(vmd),
    )
    controller = _controller(tmp_path)
    controller.add_slot(slot, model_path=pmx, motion_path=vmd)
    rest = slot.imported.skins[0].joints[1].transform.rotation.copy()
    controller.on_frame_changed(5)        # frame 5 = peak rotation in fixture
    moved = slot.imported.skins[0].joints[1].transform.rotation
    assert not np.allclose(moved, rest)


def test_run_export_vmd_writes_file(tmp_path: Path) -> None:
    from posecascade.animation.document import AnimationDocument  # noqa: PLC0415
    controller = _controller(tmp_path)
    spec = ExportSpec(
        target=ExportTarget.VMD,
        output_path=tmp_path / "out.vmd",
    )
    controller.run_export(spec, document=AnimationDocument())
    assert spec.output_path.exists()


def test_run_export_image_sequence_creates_pngs(tmp_path: Path) -> None:
    import numpy as np  # noqa: PLC0415 — late import
    controller = _controller(tmp_path)
    spec = ExportSpec(
        target=ExportTarget.IMAGE_SEQUENCE,
        output_path=tmp_path / "frames",
        start_frame=0, end_frame=2,
    )

    def render_fn(_frame: int) -> np.ndarray:
        return np.zeros((4, 4, 4), dtype=np.uint8)

    controller.run_export(spec, render_frame_fn=render_fn)
    assert sorted((tmp_path / "frames").glob("frame*.png"))


def test_run_export_vmd_without_document_raises(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    with pytest.raises(ValueError, match="VMD export"):
        controller.run_export(
            ExportSpec(target=ExportTarget.VMD, output_path=tmp_path / "x.vmd"),
        )


def test_run_export_image_sequence_without_render_fn_raises(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    with pytest.raises(ValueError, match="render_frame_fn"):
        controller.run_export(
            ExportSpec(
                target=ExportTarget.IMAGE_SEQUENCE,
                output_path=tmp_path / "frames",
            ),
        )


def test_set_effect_chain_calls_callback(tmp_path: Path) -> None:
    received: list[EffectChain] = []
    controller = AppController(
        project_root=tmp_path,
        pmx_loader=PmxImporter().load,
        vmd_loader=VmdImporter().load,
        on_chain_changed=received.append,
    )
    new_chain = EffectChain()
    controller.set_effect_chain(new_chain)
    assert received == [new_chain]


def test_save_then_open_round_trips_through_controller(tmp_path: Path) -> None:
    """End-to-end: save a project, clear state, re-open, and the same
    slot reappears with its source paths intact."""
    pmx = tmp_path / "model.pmx"
    pmx.write_bytes((_FIXTURES / "tiny.pmx").read_bytes())
    controller = _controller(tmp_path)
    controller.add_slot(_slot("character"), model_path=pmx)
    project_path = tmp_path / "demo.posecascade"
    controller.save_project_to(project_path, name="demo")

    fresh = _controller(tmp_path)
    fresh.open_project(project_path)
    reloaded = fresh.slots.find("character")
    assert reloaded is not None
    assert fresh._slot_source_paths["character"][0].name == "model.pmx"     # noqa: SLF001


def test_on_slots_changed_callback_fires_on_add(tmp_path: Path) -> None:
    """Integration glue — the MainWindow connects to this to refresh the
    slot dock after the controller mutates state."""
    received: list[SceneSlots] = []
    pmx = tmp_path / "model.pmx"
    pmx.write_bytes((_FIXTURES / "tiny.pmx").read_bytes())
    controller = AppController(
        project_root=tmp_path,
        pmx_loader=PmxImporter().load, vmd_loader=VmdImporter().load,
        on_slots_changed=received.append,
    )
    controller.add_slot(_slot("x"), model_path=pmx)
    assert received                  # at least one callback fired


# Keep ``register_builtins`` and the library types exported through this
# module so they're trivially patch-targetable in downstream tests.
__all__ = ["EffectLibrary", "register_builtins"]
