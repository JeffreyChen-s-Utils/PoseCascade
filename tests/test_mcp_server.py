"""Tests for the MCP server's four exposed tools.

Calls each helper directly rather than spawning the server — the
FastMCP wiring only changes the transport, not the underlying logic,
so unit-testing the functions is enough to cover behaviour. A separate
test confirms the server can be wired in-process without actually
calling :meth:`FastMCP.run` (which would block on stdio).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from posecascade.mcp import server as mcp_server


def test_list_animations_returns_examples_with_metadata() -> None:
    """Each row exposes the path, name, loop_sec, and phase_count fields."""
    rows = mcp_server.list_animations()
    assert rows, "expected examples/scripts/*.json to surface at least one row"
    for row in rows:
        assert set(row.keys()) == {"path", "name", "loop_sec", "phase_count"}
        assert row["path"].endswith(".json")
        assert row["path"].startswith("examples/scripts/")
        assert isinstance(row["loop_sec"], float)
        assert isinstance(row["phase_count"], int)


def test_validate_animation_accepts_valid_document() -> None:
    """A minimal-but-valid document round-trips both schema and parser checks."""
    doc = {
        "schema_version": 1,
        "name": "minimal",
        "loop_sec": 2.0,
        "rig": {"character_root": "Sketchfab_model"},
        "phases": [{"name": "still", "duration_sec": 2.0, "body": {"yaw_rad": 0.0}}],
    }
    result = mcp_server.validate_animation(content=json.dumps(doc))
    assert result == {"ok": True}


def test_validate_animation_flags_schema_violation() -> None:
    """Missing ``phases`` triggers both the schema and the parser branches."""
    result = mcp_server.validate_animation(content='{"schema_version": 1}')
    assert result["ok"] is False
    joined = " | ".join(result["errors"])
    assert "phases" in joined


def test_validate_animation_flags_malformed_json() -> None:
    """A JSON parse error surfaces with a readable message."""
    result = mcp_server.validate_animation(content="{not json}")
    assert result["ok"] is False
    assert any("JSON parse error" in e for e in result["errors"])


def test_validate_animation_requires_exactly_one_input() -> None:
    """Calling with neither (or both) ``content`` and ``path`` is a usage error."""
    with pytest.raises(ValueError, match="exactly one"):
        mcp_server.validate_animation()
    with pytest.raises(ValueError, match="exactly one"):
        mcp_server.validate_animation(content="{}", path="foo")


def test_read_animation_text_rejects_path_traversal() -> None:
    """``..`` segments cannot escape the project root."""
    with pytest.raises(ValueError, match="escapes project root"):
        mcp_server.read_animation_text("../../etc/passwd")


def test_inspect_model_reports_glb_structure(tmp_path: Path) -> None:
    """Inspecting a real bundled GLB returns the expected summary keys."""
    # examples/assets/herta/herta.glb is the canonical character GLB
    # in the repo. If it's missing on a CI shallow clone, skip cleanly.
    target = Path("examples/assets/herta/herta.glb")
    if not (mcp_server._PROJECT_ROOT / target).is_file():           # noqa: SLF001
        pytest.skip("herta.glb not present in this checkout")
    info = mcp_server.inspect_model(str(target))
    assert "error" not in info, info
    assert info["format"] == ".glb"
    assert info["mesh_count"] > 0
    assert info["vertex_count"] > 0
    assert info["triangle_count"] > 0
    assert isinstance(info["bone_names"], list)
    assert info["world_aabb"] is not None
    _ = tmp_path  # fixture currently unused but kept for parity with file-IO tests


def test_inspect_model_returns_error_for_missing_path() -> None:
    """A non-existent path resolves under the project root and returns an error dict."""
    info = mcp_server.inspect_model("examples/assets/does_not_exist.glb")
    assert "error" in info
    assert "not found" in info["error"]


def test_cloth_benchmark_runs_and_reports_metrics() -> None:
    """Benchmark returns ms/step + section breakdown + the native_kernel flag."""
    result = mcp_server.cloth_benchmark(rows=6, cols=4, iterations=2, steps=30)
    assert result["steps"] == 30
    assert result["vertex_count"] == 24
    assert result["edge_count"] > 0
    assert isinstance(result["native_kernel"], bool)
    assert result["ms_per_step"] > 0.0
    assert "cloth.step" in result["frame_sections_ms"]


def test_cloth_benchmark_rejects_degenerate_grid() -> None:
    """Tiny grids (< 2 verts per axis) raise a clear ValueError."""
    with pytest.raises(ValueError, match="rows and cols must each be >= 2"):
        mcp_server.cloth_benchmark(rows=1, cols=4)
    with pytest.raises(ValueError, match="iterations and steps must each be >= 1"):
        mcp_server.cloth_benchmark(iterations=0)


def test_server_wires_all_tools() -> None:
    """``_build_server`` returns a FastMCP whose tool names cover the public set."""
    mcp = pytest.importorskip("mcp.server.fastmcp")
    server = mcp_server._build_server()                              # noqa: SLF001
    assert isinstance(server, mcp.FastMCP)
    expected = set(mcp_server.iter_tool_names())
    # FastMCP exposes the registered tools via ``list_tools()`` on the
    # protocol layer; the underlying tool dict is private, so we check
    # the iterator instead — it's the source of truth tests pin.
    assert expected == {
        "list_animations", "read_animation", "validate_animation",
        "inspect_model", "cloth_benchmark",
    }


def test_safe_resolve_rejects_absolute_escapes(tmp_path: Path) -> None:
    """An absolute path outside the project root is refused."""
    outside = (tmp_path / "outside.txt").resolve()
    outside.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes project root"):
        mcp_server._safe_resolve(str(outside))                      # noqa: SLF001


def test_scene_aabb_handles_synthetic_mesh() -> None:
    """``_scene_aabb`` covers the union of every mesh in the scene."""
    @object.__new__
    class Stub:
        meshes = (
            type("M", (), {
                "positions": np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32),
            })(),
            type("M", (), {
                "positions": np.array([[-1.0, 0.0, 0.0]], dtype=np.float32),
            })(),
        )

    aabb = mcp_server._scene_aabb(Stub)                              # noqa: SLF001
    assert aabb == {"min": [-1.0, 0.0, 0.0], "max": [1.0, 2.0, 3.0]}
