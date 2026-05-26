"""Tests for the per-pose drape snapshot library."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from posecascade.animation.drape_snapshot import (
    SCHEMA_VERSION,
    DrapeSnapshotError,
    PoseDrapeSnapshot,
    _decode_positions,
    _encode_positions,
    apply,
    capture,
    load,
    save,
)
from posecascade.errors import MalformedAssetError


def _make_snapshot() -> PoseDrapeSnapshot:
    return PoseDrapeSnapshot(
        name="dog_crawl",
        settled_at_seconds=3.0,
        chain_states={
            "BackHairUpper": [
                (0.0, 1.7, 0.0, 0.0, 0.0, 0.0, 1.0),
                (0.05, 1.65, -0.02, 0.1, 0.0, 0.0, 0.995),
            ],
        },
        cloth_states={
            "Object_Coat": np.array(
                [[0.1, 0.5, 0.2], [-0.1, 0.5, 0.2], [0.0, 0.3, 0.4]],
                dtype=np.float32,
            ),
        },
    )


# ----- round-trip ---------------------------------------------------------


def test_to_dict_from_dict_round_trip_preserves_fields() -> None:
    snap = _make_snapshot()
    restored = PoseDrapeSnapshot.from_dict(snap.to_dict())
    assert restored.name == snap.name
    assert restored.settled_at_seconds == snap.settled_at_seconds
    assert restored.schema_version == snap.schema_version
    assert restored.chain_states == snap.chain_states
    assert set(restored.cloth_states) == set(snap.cloth_states)
    for k, v in snap.cloth_states.items():
        # float16 round-trip: tolerate ~1mm error at character scale
        np.testing.assert_allclose(restored.cloth_states[k], v, atol=2.0e-3)


def test_save_load_round_trip(tmp_path: Path) -> None:
    snap = _make_snapshot()
    out = tmp_path / "dog_crawl.drape.json"
    save(snap, out)
    assert out.exists()
    restored = load(out)
    assert restored.name == snap.name
    assert restored.chain_states == snap.chain_states


def test_save_writes_human_readable_json(tmp_path: Path) -> None:
    snap = _make_snapshot()
    out = tmp_path / "snap.json"
    save(snap, out)
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert raw["schema_version"] == SCHEMA_VERSION
    assert raw["name"] == "dog_crawl"
    assert "chain_states" in raw
    assert "cloth_states" in raw


# ----- schema validation --------------------------------------------------


def test_from_dict_rejects_wrong_schema_version() -> None:
    with pytest.raises(DrapeSnapshotError, match="schema_version"):
        PoseDrapeSnapshot.from_dict({"schema_version": 999, "name": "x"})


def test_from_dict_rejects_non_dict_input() -> None:
    with pytest.raises(DrapeSnapshotError, match="must be a dict"):
        PoseDrapeSnapshot.from_dict([1, 2, 3])  # type: ignore[arg-type]


def test_from_dict_rejects_malformed_joint_tuple() -> None:
    bad = {
        "schema_version": SCHEMA_VERSION,
        "name": "x",
        "chain_states": {"chain": [[1.0, 2.0]]},  # wrong arity
        "cloth_states": {},
    }
    with pytest.raises(DrapeSnapshotError, match="7-tuple"):
        PoseDrapeSnapshot.from_dict(bad)


def test_from_dict_rejects_malformed_chain_states_container() -> None:
    bad = {
        "schema_version": SCHEMA_VERSION,
        "name": "x",
        "chain_states": {"chain": "not a list"},
    }
    with pytest.raises(DrapeSnapshotError, match="must be a list"):
        PoseDrapeSnapshot.from_dict(bad)


def test_load_rejects_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json {", encoding="utf-8")
    with pytest.raises(MalformedAssetError, match="not valid JSON"):
        load(bad)


# ----- encode/decode helpers ---------------------------------------------


def test_encode_positions_rejects_wrong_shape() -> None:
    arr_flat = np.zeros((10,), dtype=np.float32)
    with pytest.raises(DrapeSnapshotError, match=r"\(N, 3\)"):
        _encode_positions(arr_flat)
    arr_4col = np.zeros((10, 4), dtype=np.float32)
    with pytest.raises(DrapeSnapshotError, match=r"\(N, 3\)"):
        _encode_positions(arr_4col)


def test_decode_positions_rejects_unknown_dtype() -> None:
    payload = {"dtype": "float64", "encoding": "base64+zlib", "vertex_count": 0, "data": ""}
    with pytest.raises(DrapeSnapshotError, match="unsupported cloth encoding"):
        _decode_positions(payload)


def test_decode_positions_rejects_size_mismatch() -> None:
    # Encode 3 verts but claim 99
    arr = np.zeros((3, 3), dtype=np.float32)
    payload = _encode_positions(arr)
    payload["vertex_count"] = 99
    with pytest.raises(DrapeSnapshotError, match="size mismatch"):
        _decode_positions(payload)


def test_decode_positions_rejects_bad_base64() -> None:
    payload = {"dtype": "float16", "encoding": "base64+zlib", "vertex_count": 3, "data": "!!!"}
    with pytest.raises(DrapeSnapshotError, match="base64"):
        _decode_positions(payload)


def test_encode_roundtrip_preserves_data_within_f16_tolerance() -> None:
    arr = np.random.default_rng(seed=42).standard_normal((100, 3)).astype(np.float32) * 0.5
    payload = _encode_positions(arr)
    decoded = _decode_positions(payload)
    assert decoded.shape == arr.shape
    assert decoded.dtype == np.float32
    np.testing.assert_allclose(decoded, arr, atol=2.0e-3)


def test_empty_cloth_states_round_trips() -> None:
    snap = PoseDrapeSnapshot(name="empty", chain_states={}, cloth_states={})
    restored = PoseDrapeSnapshot.from_dict(snap.to_dict())
    assert restored.chain_states == {}
    assert restored.cloth_states == {}


# ----- capture + apply with stub hosts ------------------------------------


class _StubSimulator:
    def __init__(self) -> None:
        self.snapshot_called = 0
        self.restored: dict | None = None
        self._snap = {
            "chain_a": [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)],
        }

    def snapshot_chain_state(self) -> dict:
        self.snapshot_called += 1
        return self._snap

    def restore_chain_state(self, snap: dict) -> None:
        self.restored = dict(snap)


class _StubPhysicsHost:
    def __init__(self) -> None:
        self._sim = _StubSimulator()

    @property
    def simulator(self) -> _StubSimulator:
        return self._sim


class _StubClothHost:
    def __init__(self) -> None:
        self.snapshot_called = 0
        self.restored: dict | None = None

    def snapshot_cloth_state(self) -> dict:
        self.snapshot_called += 1
        return {"piece_x": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)}

    def restore_cloth_state(self, state) -> None:
        self.restored = {k: np.asarray(v).copy() for k, v in state.items()}


def test_capture_reads_chain_and_cloth_state() -> None:
    ph = _StubPhysicsHost()
    ch = _StubClothHost()
    snap = capture(ph, ch, name="dog_crawl", settled_at_seconds=2.0)
    assert snap.name == "dog_crawl"
    assert snap.settled_at_seconds == 2.0
    assert ph.simulator.snapshot_called == 1
    assert ch.snapshot_called == 1
    assert "chain_a" in snap.chain_states
    assert "piece_x" in snap.cloth_states


def test_apply_pushes_state_back_to_hosts() -> None:
    snap = _make_snapshot()
    ph = _StubPhysicsHost()
    ch = _StubClothHost()
    apply(snap, ph, ch)
    assert ph.simulator.restored == snap.chain_states
    assert ch.restored is not None
    assert set(ch.restored) == set(snap.cloth_states)
    for k, v in snap.cloth_states.items():
        np.testing.assert_array_equal(ch.restored[k], v)


def test_apply_tolerates_hosts_without_snapshot_api() -> None:
    """Apply should no-op cleanly when hosts lack the snapshot/restore methods."""

    class _BarePhysics:  # no simulator attr
        pass

    class _BareCloth:  # no restore_cloth_state attr
        pass

    snap = _make_snapshot()
    # Must not raise.
    apply(snap, _BarePhysics(), _BareCloth())  # type: ignore[arg-type]


def test_capture_returns_empty_when_hosts_have_no_api() -> None:
    class _BarePhysics:
        pass

    class _BareCloth:
        pass

    snap = capture(_BarePhysics(), _BareCloth(), name="bare")  # type: ignore[arg-type]
    assert snap.chain_states == {}
    assert snap.cloth_states == {}
