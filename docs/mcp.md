# MCP server

PoseCascade ships a [Model Context Protocol](https://modelcontextprotocol.io/)
server so any MCP-aware LLM agent can drive the engine without going
through the desktop UI. The server is headless — it never touches Qt
or the GL context — so it runs cleanly in a subprocess over stdio.

## Install

The server lives in `posecascade.mcp.server` and ships behind the `ai`
optional dependency group:

```bash
pip install -e .[ai]
```

That pulls in `mcp` and `jsonschema`, and installs a `posecascade-mcp`
console script.

## Configure your MCP client

The repo's [`.mcp.json`](../.mcp.json) is the project-level config an
MCP-aware client (one that reads `.mcp.json` from the working tree)
will pick up automatically:

```json
{
  "$schema": "https://modelcontextprotocol.io/schema/server-config.json",
  "mcpServers": {
    "posecascade": {
      "command": "posecascade-mcp",
      "args": [],
      "env": {}
    }
  }
}
```

For the `posecascade-mcp` command to resolve, the venv that installed
PoseCascade with the `ai` extra has to be on PATH. The simplest pattern
is to **activate the venv before launching the client**:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
claude               # or your other MCP client
```

If you'd rather not rely on PATH, point the config at the venv-resolved
script directly:

```json
{
  "mcpServers": {
    "posecascade": {
      "command": ".venv/Scripts/posecascade-mcp.exe"
    }
  }
}
```

(Linux / macOS: `.venv/bin/posecascade-mcp`.)

## Tools

The server exposes five tools. All are read-only or pure compute; none
mutates the working tree.

### `list_animations`

Returns every `.json` declarative animation under `examples/scripts/`
with its `name`, `loop_sec`, and phase count.

```json
[
  {
    "path": "examples/scripts/walk.json",
    "name": "walk_in_place_demo",
    "loop_sec": 4.0,
    "phase_count": 1
  }
]
```

Use it to discover what's available before writing a new animation —
the existing scripts are the best reference for the schema.

### `read_animation`

Returns the raw JSON text of one of those files. Pass a project-
relative path (path traversal — `..` — is rejected).

### `validate_animation`

JSON-Schema check **plus** runtime-parser check. Pass exactly one of
`content` (inline JSON text) or `path` (project-relative).

```json
// Success
{"ok": true}

// Failure
{
  "ok": false,
  "errors": [
    "<root>: 'phases' is a required property",
    "parse error: animation must have at least one phase"
  ]
}
```

The schema engine catches structural issues (missing required fields,
wrong types); the parser catches semantic ones (unknown bones,
malformed expressions, conflicting `duration_sec` / `duration_beats`).

### `inspect_model`

Imports any supported model file (`.glb`, `.gltf`, `.pmx`, `.pmd`,
`.obj`, `.stl`, `.ply`, `.fbx`, `.usd`, `.usdz`, `.dae`) and returns a
structural summary:

```json
{
  "path": "examples/assets/herta/herta.glb",
  "format": ".glb",
  "mesh_count": 12,
  "texture_count": 15,
  "skin_count": 1,
  "node_count": 154,
  "vertex_count": 14509,
  "triangle_count": 20808,
  "bone_names": ["_rootJoint", "Root", "J_Bip_C_Hips", "..."],
  "material_names": [],
  "world_aabb": {"min": [-0.64, -0.13, -1.55], "max": [0.63, 0.97, 0.13]}
}
```

The bone-name list is capped at 20 entries so the response stays compact.
On import failure (corrupt file, unsupported extension, missing
dependency) the tool returns `{"error": "..."}` rather than raising.

### `cloth_benchmark`

Builds a synthetic `rows × cols` grid, anchors the top row, drops
gravity + a sphere collider, and runs the cloth solver for `steps`
iterations. Returns ms/step (best of three runs after a 100-step
warmup), the `frame_section` breakdown, and crucially the
`native_kernel` flag — `true` means the Cython kernel is loaded,
`false` means the NumPy fallback is running.

```json
{
  "ms_per_step": 0.358,
  "total_ms": 215.0,
  "steps": 600,
  "vertex_count": 480,
  "edge_count": 1337,
  "bend_count": 1237,
  "iterations_per_step": 8,
  "native_kernel": true,
  "frame_sections_ms": {"cloth.step": 213.5}
}
```

Use it to:

- Verify the Cython kernel loaded after a `pip install` (the
  `native_kernel` flag is the canary).
- Validate a perf change end-to-end — the defaults match the numbers
  cited in the [README](../README.md#performance-notes), so any drift
  is immediately visible.
- Spot-check ABI mismatches when distributing pre-built wheels.

## Path safety

Both `read_animation` and `inspect_model` resolve their argument via
`_safe_resolve`, which rejects:

- Any path whose resolved location escapes the project root (so
  `../../etc/passwd` and absolute paths outside the tree both fail).
- Missing files (returned as an error dict rather than a raised
  exception, so the agent can recover gracefully).

This mirrors the same hardening
`posecascade.assets.path_safety.resolve_safe` applies to model-file
URIs at runtime — the MCP entry points are not a privilege-escalation
hole.

## Adding a new tool

1. Add a pure function to `posecascade/mcp/server.py` next to the
   other tool implementations. Keep it headless (no Qt, no GL).
2. Register it in `_build_server` with a one-line `description=` that
   spells out the contract for the calling agent.
3. Add the name to `iter_tool_names()` — the test
   `test_server_wires_all_tools` pins the public surface.
4. Write a unit test in `tests/test_mcp_server.py` that calls the
   helper directly with synthetic inputs. The transport layer is
   FastMCP's responsibility; testing your code's behaviour at the
   function level is enough.
