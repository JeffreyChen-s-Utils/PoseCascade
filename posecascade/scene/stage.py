"""Procedural stage geometry.

A *stage* is a passive prop the dancer stands on: a raised platform,
walls, no skinning, no animation. We ship a built-in procedural stage
so the editor has something MMD-ish to drop a model onto even before
the user imports a real ``.pmx`` / ``.glb`` stage file. The geometry
is plain enough to land in an
:class:`~posecascade.assets.types.ImportedScene` that the existing
renderer / slot machinery can consume unchanged.

The procedural stage is four meshes — floor + back wall + two side
walls — forming a "stage box" the dancer occupies:

- A 5 m × 5 m raised dance floor at ``y = 0.05`` with a light tint so
  it reads distinct from the procedural checkered ground beneath it.
- A 5 m wide × 3 m tall back wall at ``z = -2.5``.
- Two 3 m deep × 3 m tall side walls at ``x = ±2.5`` running from
  ``z = -2.5`` to ``z = 0.5``.

Replace this procedural stage with any imported PMX / glTF scene to
load a real stage asset:

.. code-block:: python

    from posecascade.animation.slots_player import make_stage_slot
    from posecascade.assets.importer_manager import ImporterManager

    manager = ImporterManager(importers_root=...)
    manager.discover()
    imported_stage = manager.load(Path("path/to/stage.pmx"))
    slots.add(make_stage_slot(name="theatre", imported=imported_stage))

The slot machinery doesn't care what's inside ``imported`` — only the
renderer's mesh walk consumes the geometry, and the slots player
skips per-frame animation passes on ``is_stage=True`` slots.
"""
from __future__ import annotations

import numpy as np

from posecascade.assets.types import ImportedScene, Mesh
from posecascade.scene.component import MeshRefComponent
from posecascade.scene.node import Node
from posecascade.scene.scene import Scene

_FLOOR_HALF_SIDE = 2.5
_FLOOR_Y = 0.05
_FLOOR_COLOR = (0.85, 0.88, 0.93, 1.0)
_BACK_WALL_HALF_WIDTH = 2.5
_BACK_WALL_HEIGHT = 3.0
_BACK_WALL_Z = -2.5
_BACK_WALL_COLOR = (0.62, 0.66, 0.74, 1.0)
# Side walls run forward from the back wall to slightly past the front
# of the dance floor. Slightly darker tint than the back wall so the
# camera has a depth cue.
_SIDE_WALL_X = 2.5
_SIDE_WALL_Z_BACK = -2.5
_SIDE_WALL_Z_FRONT = 0.5
_SIDE_WALL_HEIGHT = 3.0
_SIDE_WALL_COLOR = (0.55, 0.59, 0.66, 1.0)


def _quad_mesh(
    name: str,
    p00: tuple[float, float, float],
    p10: tuple[float, float, float],
    p11: tuple[float, float, float],
    p01: tuple[float, float, float],
    normal: tuple[float, float, float],
    color: tuple[float, float, float, float],
) -> Mesh:
    """Two triangles in CCW order with shared per-vertex normal + base colour."""
    positions = np.array([p00, p10, p11, p01], dtype=np.float32)
    normals = np.tile(np.array(normal, dtype=np.float32), (4, 1))
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    return Mesh(
        name=name,
        positions=positions,
        normals=normals,
        indices=indices,
        base_color=color,
    )


def procedural_dance_stage() -> ImportedScene:
    """Build the bundled procedural stage as an :class:`ImportedScene`.

    Four unskinned meshes (floor + back wall + two side walls) under a
    fresh :class:`Scene` root. The result drops into the existing slot
    machinery — register it via :func:`make_stage_slot`.
    """
    floor = _quad_mesh(
        name="stage_floor",
        p00=(-_FLOOR_HALF_SIDE, _FLOOR_Y, -_FLOOR_HALF_SIDE),
        p10=( _FLOOR_HALF_SIDE, _FLOOR_Y, -_FLOOR_HALF_SIDE),
        p11=( _FLOOR_HALF_SIDE, _FLOOR_Y,  _FLOOR_HALF_SIDE),
        p01=(-_FLOOR_HALF_SIDE, _FLOOR_Y,  _FLOOR_HALF_SIDE),
        normal=(0.0, 1.0, 0.0),
        color=_FLOOR_COLOR,
    )
    back_wall = _quad_mesh(
        name="stage_wall",
        p00=(-_BACK_WALL_HALF_WIDTH, 0.0,                _BACK_WALL_Z),
        p10=( _BACK_WALL_HALF_WIDTH, 0.0,                _BACK_WALL_Z),
        p11=( _BACK_WALL_HALF_WIDTH, _BACK_WALL_HEIGHT,  _BACK_WALL_Z),
        p01=(-_BACK_WALL_HALF_WIDTH, _BACK_WALL_HEIGHT,  _BACK_WALL_Z),
        normal=(0.0, 0.0, 1.0),
        color=_BACK_WALL_COLOR,
    )
    # Left side wall — normal points +X (into the stage interior).
    left_wall = _quad_mesh(
        name="stage_wall_left",
        p00=(-_SIDE_WALL_X, 0.0,                _SIDE_WALL_Z_BACK),
        p10=(-_SIDE_WALL_X, 0.0,                _SIDE_WALL_Z_FRONT),
        p11=(-_SIDE_WALL_X, _SIDE_WALL_HEIGHT,  _SIDE_WALL_Z_FRONT),
        p01=(-_SIDE_WALL_X, _SIDE_WALL_HEIGHT,  _SIDE_WALL_Z_BACK),
        normal=(1.0, 0.0, 0.0),
        color=_SIDE_WALL_COLOR,
    )
    # Right side wall — normal points -X (into the stage interior). The
    # CCW order from the inside is the reverse of the left wall so back-
    # face culling still treats the visible side as the front.
    right_wall = _quad_mesh(
        name="stage_wall_right",
        p00=( _SIDE_WALL_X, 0.0,                _SIDE_WALL_Z_FRONT),
        p10=( _SIDE_WALL_X, 0.0,                _SIDE_WALL_Z_BACK),
        p11=( _SIDE_WALL_X, _SIDE_WALL_HEIGHT,  _SIDE_WALL_Z_BACK),
        p01=( _SIDE_WALL_X, _SIDE_WALL_HEIGHT,  _SIDE_WALL_Z_FRONT),
        normal=(-1.0, 0.0, 0.0),
        color=_SIDE_WALL_COLOR,
    )

    scene = Scene(name="procedural_stage")
    for index, name in enumerate(
        ("stage_floor", "stage_wall", "stage_wall_left", "stage_wall_right"),
    ):
        node = Node(name=name)
        node.add_component(MeshRefComponent(mesh_indices=(index,)))
        scene.root.add_child(node)
    return ImportedScene(
        meshes=(floor, back_wall, left_wall, right_wall),
        scene=scene,
    )
