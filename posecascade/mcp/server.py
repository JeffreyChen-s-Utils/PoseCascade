"""MCP server exposing PoseCascade introspection + benchmarking tools.

Run via the ``posecascade-mcp`` console script (installed by ``pip install
-e .``) or directly with ``python -m posecascade.mcp.server``. The server
speaks the Model Context Protocol over stdio — the project's
``.mcp.json`` already points at it so any MCP-aware client picks it up
automatically.

The four tools are deliberately read-only or pure compute. None of them
touches the GL renderer or Qt event loop, so the server runs cleanly in
a headless subprocess.

- :func:`list_animations` — enumerate every declarative animation under
  ``examples/scripts/``.
- :func:`validate_animation` — JSON-Schema check + runtime-parser check
  against the bundled v1 schema.
- :func:`inspect_model` — import any supported model file (glTF/GLB,
  PMX, PMD, OBJ, STL, PLY, FBX, USD, DAE) and return a structural
  summary.
- :func:`cloth_benchmark` — synthesize a grid skirt, run the cloth
  solver for N steps, and report a ms/step breakdown.

A separate ``main()`` entrypoint binds the tools to a :class:`FastMCP`
instance and spins up the stdio transport. Tests import the underlying
helpers directly to avoid spawning a subprocess in unit-test scope.
"""
from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Lazily importable because the server may be invoked from a tree that
# doesn't yet have the importer plugins on sys.path. ``_ensure_paths``
# below adjusts sys.path the way the editor app does at startup.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_IMPORTERS_ROOT = _PROJECT_ROOT / "importers"


def _ensure_paths() -> None:
    """Mirror the runtime ``sys.path`` adjustment the editor app performs.

    The importer plugins live outside the ``posecascade`` package on
    purpose (see :mod:`posecascade.assets.importer_manager`), and the
    MCP server needs the same prepend before it can call
    :func:`ImporterManager.discover`.
    """
    if str(_IMPORTERS_ROOT) not in sys.path:
        sys.path.insert(0, str(_IMPORTERS_ROOT))


_SCHEMA_PATH = _PROJECT_ROOT / "schemas" / "animation_v1.json"
_EXAMPLES_DIR = _PROJECT_ROOT / "examples" / "scripts"
_ASSETS_DIR = _PROJECT_ROOT / "examples" / "assets"


# ---------------------------------------------------------------------------
# list_animations
# ---------------------------------------------------------------------------


@dataclass
class AnimationSummary:
    """One row of :func:`list_animations`' result."""

    path: str            # path relative to project root
    name: str            # ``name`` field from the JSON, or ``""``
    loop_sec: float      # ``loop_sec`` field, or 0
    phase_count: int     # number of phases declared


def list_animations() -> list[dict[str, Any]]:
    """List every ``.json`` declarative animation under ``examples/scripts/``.

    The result is a list of dicts with ``path``, ``name``, ``loop_sec``,
    and ``phase_count``. Pass any returned ``path`` to
    :func:`validate_animation` (with ``path=``) to get a parser-level
    check, or feed it back to :func:`read_animation_text` to inspect the
    raw document.
    """
    rows: list[dict[str, Any]] = []
    if not _EXAMPLES_DIR.is_dir():
        return rows
    for json_path in sorted(_EXAMPLES_DIR.glob("*.json")):
        try:
            doc = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(
            {
                "path": str(json_path.relative_to(_PROJECT_ROOT)).replace("\\", "/"),
                "name": str(doc.get("name", "")),
                "loop_sec": float(doc.get("loop_sec", 0.0)),
                "phase_count": len(doc.get("phases", [])),
            },
        )
    return rows


def read_animation_text(path: str) -> str:
    """Return the raw JSON text of ``path`` (relative to the project root).

    Guards path traversal: the resolved path must sit under the project
    root, so the agent can't read arbitrary disk locations through this
    tool.
    """
    resolved = _safe_resolve(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"animation not found: {path}")
    return resolved.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# validate_animation
# ---------------------------------------------------------------------------


def validate_animation(
    *, content: str | None = None, path: str | None = None,
) -> dict[str, Any]:
    """Validate a declarative animation document.

    Pass exactly one of ``content`` (raw JSON text) or ``path`` (project-
    relative). Returns ``{"ok": True}`` on full success or
    ``{"ok": False, "errors": [...]}`` listing every schema + parser
    failure. Both engines run: the JSON-Schema validator catches
    structural issues, then the runtime parser catches semantic ones
    (unknown bones, malformed expressions, …).
    """
    if (content is None) == (path is None):
        raise ValueError("pass exactly one of content= or path=")
    text = content if content is not None else read_animation_text(path or "")
    errors: list[str] = []
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        return {"ok": False, "errors": [f"JSON parse error: {exc}"]}
    errors.extend(_schema_errors(doc))
    errors.extend(_parser_errors(doc))
    if errors:
        return {"ok": False, "errors": errors}
    return {"ok": True}


def _schema_errors(doc: dict[str, Any]) -> list[str]:
    """Return every JSON-Schema violation for ``doc``."""
    if not _SCHEMA_PATH.is_file():
        return []
    try:
        import jsonschema  # noqa: PLC0415 — optional dep, only loaded for the MCP tool
    except ImportError:
        return []
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"{'/'.join(str(p) for p in error.absolute_path) or '<root>'}: {error.message}"
        for error in validator.iter_errors(doc)
    ]


def _parser_errors(doc: dict[str, Any]) -> list[str]:
    """Return the runtime parser's complaint, if any."""
    from posecascade.scripting.declarative import (  # noqa: PLC0415 — optional dep
        DeclarativeAnimationError,
        parse_animation,
    )

    try:
        parse_animation(doc)
    except DeclarativeAnimationError as exc:
        return [f"parse error: {exc}"]
    return []


# ---------------------------------------------------------------------------
# inspect_model
# ---------------------------------------------------------------------------


def inspect_model(path: str) -> dict[str, Any]:
    """Import a model and return a structural summary.

    Supports every format an importer plugin is registered for (glTF,
    GLB, PMX, PMD, OBJ, STL, PLY, FBX, USD, DAE). Returns mesh / texture
    counts, total vertex + triangle counts, the first few bone names
    when a skin is present, material names, and the world-space AABB
    of every mesh combined.

    Heavy file? Returns ``{"error": "..."}`` if the importer raises;
    the agent can recover by trying a smaller file or a different
    format.
    """
    _ensure_paths()
    resolved = _safe_resolve(path)
    if not resolved.is_file():
        return {"error": f"model not found: {path}"}
    from posecascade.assets.importer_manager import (  # noqa: PLC0415
        ImporterManager,
    )

    manager = ImporterManager(importers_root=_IMPORTERS_ROOT)
    manager.discover()
    try:
        scene = manager.load(resolved)
    except Exception as exc:                            # noqa: BLE001 — surface any importer error
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "path": str(resolved.relative_to(_PROJECT_ROOT)).replace("\\", "/"),
        "format": resolved.suffix.lower(),
        "mesh_count": len(scene.meshes),
        "texture_count": len(scene.textures),
        "skin_count": len(scene.skins),
        "node_count": _count_nodes(scene.scene) if scene.scene is not None else 0,
        "vertex_count": sum(int(m.positions.shape[0]) for m in scene.meshes),
        "triangle_count": sum(int(m.indices.size // 3) for m in scene.meshes),
        "bone_names": _first_bone_names(scene),
        "material_names": _material_names(scene.meshes),
        "world_aabb": _scene_aabb(scene),
    }


def _count_nodes(scene: Any) -> int:
    return sum(1 for _ in scene.root.traverse())


def _first_bone_names(scene: Any, limit: int = 20) -> list[str]:
    """First ``limit`` joint names from the first skin — full lists get noisy."""
    if not scene.skins:
        return []
    return [j.name for j in scene.skins[0].joints[:limit]]


def _material_names(meshes: Any) -> list[str]:
    """Unique material names across every mesh."""
    seen: list[str] = []
    for mesh in meshes:
        material = getattr(mesh, "mmd_material", None)
        if material is None:
            continue
        if material.name not in seen:
            seen.append(material.name)
    return seen


def _scene_aabb(scene: Any) -> dict[str, list[float]] | None:
    """Union of every mesh's vertex AABB. Returns ``None`` if there are no meshes."""
    if not scene.meshes:
        return None
    positions = np.concatenate(
        [np.asarray(m.positions, dtype=np.float32) for m in scene.meshes], axis=0,
    )
    return {
        "min": [float(x) for x in positions.min(axis=0)],
        "max": [float(x) for x in positions.max(axis=0)],
    }


# ---------------------------------------------------------------------------
# cloth_benchmark
# ---------------------------------------------------------------------------


_DEFAULT_BENCH_DT = 1.0 / 60.0
# A grid with fewer than two verts per axis can't form a single quad,
# so the cloth solver has nothing to constrain. Reject early with a
# clear error rather than failing inside ``cloth_from_mesh``.
_MIN_GRID_SIDE = 2


def cloth_benchmark(
    rows: int = 20,
    cols: int = 10,
    iterations: int = 8,
    steps: int = 200,
) -> dict[str, Any]:
    """Run the cloth solver on a synthetic skirt and report timing.

    Builds a ``rows × cols`` grid, anchors the top row, drops gravity +
    a single sphere collider near the middle, and steps the solver
    ``steps`` times. Returns ms/step (best of 3 runs after a 100-step
    warmup), total vert / edge / bend counts, and the ``frame_section``
    breakdown. The ``native_kernel`` flag reports whether the compiled
    Cython kernel is loaded — useful for catching ABI mismatches when
    distributing pre-built wheels.

    The defaults match the bench numbers cited in the perf docs so the
    agent can re-verify them after a change.
    """
    if rows < _MIN_GRID_SIDE or cols < _MIN_GRID_SIDE:
        raise ValueError("rows and cols must each be >= 2")
    if iterations < 1 or steps < 1:
        raise ValueError("iterations and steps must each be >= 1")
    from posecascade.animation import cloth as cloth_mod  # noqa: PLC0415
    from posecascade.animation.cloth import (  # noqa: PLC0415
        ClothGravity,
        ClothParams,
        ClothSolver,
        SphereCollider,
        cloth_from_mesh,
    )
    from posecascade.utils.math3d import mat4_identity  # noqa: PLC0415
    from posecascade.utils.profiling import current_stats  # noqa: PLC0415

    positions, indices, anchor_mask = _build_grid(rows, cols)
    piece = cloth_from_mesh(
        name="bench",
        local_positions=positions,
        indices=indices,
        world_matrix=mat4_identity(),
        anchor_mask=anchor_mask,
        params=ClothParams(iterations=iterations),
    )
    solver = ClothSolver(pieces=[piece])
    solver.forces.append(ClothGravity())
    solver.colliders.append(
        SphereCollider(
            center=np.array([cols * 0.04 * 0.5, 0.5, 0.0], dtype=np.float32),
            radius=0.08,
        ),
    )
    for _ in range(100):
        solver.step(_DEFAULT_BENCH_DT)
    current_stats().reset()
    elapsed = min(_time_run(solver, steps) for _ in range(3))
    sections = dict(current_stats().sections)
    return {
        "ms_per_step": elapsed * 1000.0 / steps,
        "total_ms": elapsed * 1000.0,
        "steps": steps,
        "vertex_count": int(piece.positions.shape[0]),
        "edge_count": int(piece.edges.shape[0]),
        "bend_count": int(piece.bends.shape[0]),
        "iterations_per_step": iterations,
        "native_kernel": cloth_mod._native is not None,             # noqa: SLF001
        "frame_sections_ms": {name: float(ms) for name, ms in sections.items()},
    }


def _time_run(solver: Any, steps: int) -> float:
    """Wall-clock for ``steps`` solver iterations."""
    t0 = time.perf_counter()
    for _ in range(steps):
        solver.step(_DEFAULT_BENCH_DT)
    return time.perf_counter() - t0


def _build_grid(
    rows: int, cols: int, spacing: float = 0.04,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(positions, indices, anchor_mask)`` for a flat XY grid."""
    positions = np.zeros((rows * cols, 3), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            positions[r * cols + c] = (c * spacing, 1.0 - r * spacing, 0.0)
    indices: list[int] = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            a = r * cols + c
            b = a + 1
            c0 = a + cols
            d = c0 + 1
            indices += [a, c0, b, b, c0, d]
    anchor_mask = np.zeros(rows * cols, dtype=bool)
    anchor_mask[:cols] = True
    return positions, np.asarray(indices, dtype=np.uint32), anchor_mask


# ---------------------------------------------------------------------------
# path safety
# ---------------------------------------------------------------------------


def _safe_resolve(relative: str) -> Path:
    """Resolve ``relative`` under the project root, rejecting traversal.

    Mirrors :func:`posecascade.assets.path_safety.resolve_safe` but for
    the MCP entry points so the agent cannot read arbitrary disk
    locations via these tools.
    """
    candidate = (_PROJECT_ROOT / relative).resolve()
    try:
        candidate.relative_to(_PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes project root: {relative}") from exc
    return candidate


# ---------------------------------------------------------------------------
# server entry point
# ---------------------------------------------------------------------------


def _build_server() -> Any:
    """Wire the tools into a fresh :class:`FastMCP` instance.

    Imported lazily so unit tests can call the underlying helpers
    without forcing the ``mcp`` dependency onto every contributor.
    """
    from mcp.server.fastmcp import FastMCP  # noqa: PLC0415

    server = FastMCP("posecascade")
    server.tool(
        name="list_animations",
        description=(
            "Enumerate every declarative animation under examples/scripts/. "
            "Returns path, name, loop_sec, and phase_count for each .json file."
        ),
    )(list_animations)
    server.tool(
        name="read_animation",
        description=(
            "Return the raw JSON text of a declarative animation document. "
            "Use a path returned by list_animations, or any .json under "
            "examples/scripts/."
        ),
    )(read_animation_text)
    server.tool(
        name="validate_animation",
        description=(
            "Validate a declarative animation against the v1 JSON Schema + "
            "the runtime parser. Pass content= (inline JSON text) or path= "
            "(relative to project root) — exactly one. Returns {ok: true} or "
            "{ok: false, errors: [...]}."
        ),
    )(validate_animation)
    server.tool(
        name="inspect_model",
        description=(
            "Import a 3D model and return a structural summary: mesh / texture "
            "/ skin counts, total verts and triangles, first 20 bone names, "
            "material names, and the world-space AABB. Supported formats: "
            ".glb .gltf .obj .stl .ply .pmx .pmd .fbx .usd .usdz .dae."
        ),
    )(inspect_model)
    server.tool(
        name="cloth_benchmark",
        description=(
            "Run the cloth solver on a synthetic grid and return ms/step + "
            "per-section breakdown. native_kernel: true means the compiled "
            "Cython kernel is loaded; false falls back to NumPy."
        ),
    )(cloth_benchmark)
    return server


def main() -> int:
    """Start the stdio server. Wired to the ``posecascade-mcp`` console script."""
    _ensure_paths()
    server = _build_server()
    server.run()
    return 0


def iter_tool_names() -> Iterator[str]:
    """Yield every tool name registered by :func:`_build_server`.

    Exists so tests can pin the public surface without spawning the
    server.
    """
    yield from (
        "list_animations",
        "read_animation",
        "validate_animation",
        "inspect_model",
        "cloth_benchmark",
    )


if __name__ == "__main__":                              # pragma: no cover
    sys.exit(main())
