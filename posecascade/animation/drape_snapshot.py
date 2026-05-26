"""Per-pose drape snapshot library.

A drape snapshot captures the settled visual state of every spring chain
and PBD cloth piece for one specific pose. Loading the snapshot at
runtime skips warmup and locks the visuals into the authored shape —
the same trick HoYoverse uses for *Genshin* / *Honkai: Star Rail* prone
and kneeling animations, and Magica Cloth ships as its "Pose Snapshot"
feature.

Use it when physics simulation cannot reliably reach the desired shape
on its own: extreme stances (dog-crawl, hand-stand, prone), cinematics
where determinism matters more than micro-motion, or when an asset's
bind-pose geometry fights gravity in the target pose.

Workflow:

1. Run the scene to settle into the pose (warmup + a few seconds of sim).
2. Call :func:`capture` on the live physics + cloth hosts to grab a
   :class:`PoseDrapeSnapshot`.
3. Call :func:`save` to write a versioned JSON file under ``examples/poses/``.
4. At runtime, declare ``"drape_snapshot": "examples/poses/<name>.drape.json"``
   on a phase; the declarative runtime loads + applies it at phase start.

The snapshot stores **world-space** joint poses and cloth vertex positions.
That's fine for stationary cinematics (single character at origin); for
characters that move, prefer scripted spring forces or re-bake per
checkpoint.
"""
from __future__ import annotations

import base64
import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from posecascade.errors import MalformedAssetError, PoseCascadeError

if TYPE_CHECKING:
    from posecascade.animation.cloth_host import ClothHost
    from posecascade.animation.physics_host import PhysicsHost


SCHEMA_VERSION = 1
# Float16 quantisation cuts cloth-state payload in half with sub-millimetre
# error at typical character scales — visually indistinguishable from
# float32 once the renderer has smoothed normals and applied lighting.
_CLOTH_DTYPE = np.float16
_VEC3_LEN = 3
_QUAT_TUPLE_LEN = 7  # (px, py, pz, qx, qy, qz, qw)
_VERT_ARRAY_NDIM = 2  # (N, 3) vertex arrays are 2-D


class DrapeSnapshotError(PoseCascadeError):
    """A drape snapshot file is malformed or incompatible."""


@dataclass
class PoseDrapeSnapshot:
    """Settled drape state for one pose.

    ``chain_states`` maps a spring-chain name to a list of per-joint
    ``(px, py, pz, qx, qy, qz, qw)`` tuples (world position + world
    rotation). ``cloth_states`` maps a cloth-piece name to its (N, 3)
    world-space vertex positions.

    Snapshots are immutable in spirit — once captured, only serialise
    or apply them. Use :func:`capture` to build a fresh one.
    """

    name: str
    settled_at_seconds: float = 0.0
    chain_states: dict[str, list[tuple[float, float, float, float, float, float, float]]] = field(
        default_factory=dict,
    )
    cloth_states: dict[str, NDArray[np.float32]] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for ``json.dump``."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "settled_at_seconds": self.settled_at_seconds,
            "chain_states": {
                chain: [list(t) for t in joints]
                for chain, joints in self.chain_states.items()
            },
            "cloth_states": {
                name: _encode_positions(arr) for name, arr in self.cloth_states.items()
            },
        }

    @classmethod
    def from_dict(cls, raw: dict) -> PoseDrapeSnapshot:
        """Parse a snapshot from the dict produced by :meth:`to_dict`."""
        if not isinstance(raw, dict):
            raise DrapeSnapshotError(f"snapshot must be a dict, got {type(raw).__name__}")
        version = raw.get("schema_version")
        if version != SCHEMA_VERSION:
            raise DrapeSnapshotError(
                f"unsupported schema_version {version!r}; this build only "
                f"accepts {SCHEMA_VERSION}",
            )
        chain_states: dict[str, list[tuple[float, ...]]] = {}
        for chain, joints in (raw.get("chain_states") or {}).items():
            if not isinstance(joints, list):
                raise DrapeSnapshotError(
                    f"chain_states[{chain!r}] must be a list of joint tuples",
                )
            parsed: list[tuple[float, ...]] = []
            for joint in joints:
                if not isinstance(joint, (list, tuple)) or len(joint) != _QUAT_TUPLE_LEN:
                    raise DrapeSnapshotError(
                        f"chain_states[{chain!r}] joint must be a "
                        f"{_QUAT_TUPLE_LEN}-tuple (px, py, pz, qx, qy, qz, qw)",
                    )
                parsed.append(tuple(float(v) for v in joint))
            chain_states[str(chain)] = parsed
        cloth_states: dict[str, NDArray[np.float32]] = {}
        for name, payload in (raw.get("cloth_states") or {}).items():
            cloth_states[str(name)] = _decode_positions(payload)
        return cls(
            schema_version=int(version),
            name=str(raw.get("name", "")),
            settled_at_seconds=float(raw.get("settled_at_seconds", 0.0)),
            chain_states=chain_states,
            cloth_states=cloth_states,
        )


def capture(
    physics_host: PhysicsHost,
    cloth_host: ClothHost,
    name: str,
    settled_at_seconds: float = 0.0,
) -> PoseDrapeSnapshot:
    """Capture the live drape state of every chain + cloth piece.

    The hosts must have already ticked enough to settle into the target
    pose — typically 60-180 frames of warmup with the pose's bone overrides
    applied. The capture is a pure read; it does not mutate the hosts.
    """
    chain_states = _capture_chain_states(physics_host)
    cloth_states = _capture_cloth_states(cloth_host)
    return PoseDrapeSnapshot(
        name=name,
        settled_at_seconds=float(settled_at_seconds),
        chain_states=chain_states,
        cloth_states=cloth_states,
    )


def apply(
    snapshot: PoseDrapeSnapshot,
    physics_host: PhysicsHost,
    cloth_host: ClothHost,
    freeze: bool = False,
) -> None:
    """Apply a snapshot to the live hosts. Chains and pieces not present in
    the snapshot are left running normally.

    When ``freeze`` is ``True`` (default), the spring chains and cloth pieces
    that the snapshot restored are switched to ``enabled=False`` so per-tick
    integration stops touching them — the pose is locked at the authored
    drape just like HoYoverse's per-pose static drape locks Genshin /
    Honkai cinematics. Pass ``freeze=False`` to keep simulation running
    on top of the restored state (useful for "warmup skip" workflows
    where you want the snapshot to be the initial condition rather than
    a permanent override).
    """
    if snapshot.chain_states and hasattr(physics_host, "simulator"):
        sim = physics_host.simulator
        sim.restore_chain_state(snapshot.chain_states)
        if freeze and hasattr(sim, "chains"):
            for chain in sim.chains:
                if chain.name in snapshot.chain_states:
                    chain.frozen = True
    if snapshot.cloth_states and hasattr(cloth_host, "restore_cloth_state"):
        cloth_host.restore_cloth_state(snapshot.cloth_states)
        if freeze and hasattr(cloth_host, "_solver"):
            solver = cloth_host._solver  # noqa: SLF001  # private-by-convention API
            for piece in solver.pieces:
                if piece.name in snapshot.cloth_states:
                    piece.frozen = True


def save(snapshot: PoseDrapeSnapshot, path: Path) -> None:
    """Write a snapshot as JSON at ``path``. Parent dir must exist."""
    payload = snapshot.to_dict()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load(path: Path) -> PoseDrapeSnapshot:
    """Read a snapshot from a JSON file. Raises :class:`DrapeSnapshotError`
    on malformed content and :class:`MalformedAssetError` on invalid JSON.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise MalformedAssetError(f"drape snapshot {path!s} is not valid JSON") from err
    return PoseDrapeSnapshot.from_dict(raw)


def _capture_chain_states(
    physics_host: PhysicsHost,
) -> dict[str, list[tuple[float, float, float, float, float, float, float]]]:
    if not hasattr(physics_host, "simulator"):
        return {}
    sim = physics_host.simulator
    if sim is None or not hasattr(sim, "snapshot_chain_state"):
        return {}
    return sim.snapshot_chain_state()


def _capture_cloth_states(cloth_host: ClothHost) -> dict[str, NDArray[np.float32]]:
    if not hasattr(cloth_host, "snapshot_cloth_state"):
        return {}
    return cloth_host.snapshot_cloth_state()


def _encode_positions(positions: NDArray[np.float32]) -> dict[str, object]:
    """Pack an (N, 3) float array as base64-encoded, zlib-compressed float16."""
    arr = np.ascontiguousarray(positions, dtype=_CLOTH_DTYPE)
    if arr.ndim != _VERT_ARRAY_NDIM or arr.shape[1] != _VEC3_LEN:
        raise DrapeSnapshotError(
            f"cloth positions must be (N, 3); got shape {arr.shape}",
        )
    raw = zlib.compress(arr.tobytes(), level=6)
    return {
        "vertex_count": int(arr.shape[0]),
        "dtype": "float16",
        "encoding": "base64+zlib",
        "data": base64.b64encode(raw).decode("ascii"),
    }


def _decode_positions(payload: object) -> NDArray[np.float32]:
    """Inverse of :func:`_encode_positions`; returns an (N, 3) float32 array."""
    if not isinstance(payload, dict):
        raise DrapeSnapshotError(
            f"cloth_states entry must be a dict, got {type(payload).__name__}",
        )
    if payload.get("dtype") != "float16" or payload.get("encoding") != "base64+zlib":
        raise DrapeSnapshotError(
            f"unsupported cloth encoding: dtype={payload.get('dtype')!r} "
            f"encoding={payload.get('encoding')!r}",
        )
    vertex_count = int(payload.get("vertex_count", 0))
    raw = payload.get("data")
    if not isinstance(raw, str):
        raise DrapeSnapshotError("cloth_states.data must be a base64 string")
    try:
        compressed = base64.b64decode(raw, validate=True)
    except (ValueError, TypeError) as err:
        raise DrapeSnapshotError(f"cloth_states.data base64 decode failed: {err}") from err
    try:
        decompressed = zlib.decompress(compressed)
    except zlib.error as err:
        raise DrapeSnapshotError(f"cloth_states.data zlib decompress failed: {err}") from err
    expected_bytes = vertex_count * _VEC3_LEN * np.dtype(_CLOTH_DTYPE).itemsize
    if len(decompressed) != expected_bytes:
        raise DrapeSnapshotError(
            f"cloth_states payload size mismatch: got {len(decompressed)} bytes, "
            f"expected {expected_bytes} for {vertex_count} verts",
        )
    arr_f16 = np.frombuffer(decompressed, dtype=_CLOTH_DTYPE).reshape((vertex_count, _VEC3_LEN))
    return arr_f16.astype(np.float32, copy=True)
