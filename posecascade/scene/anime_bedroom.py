"""Procedural anime-style bedroom — boxes for furniture, soft pastel palette.

Returns an :class:`~posecascade.assets.types.ImportedScene` so the standard
``Renderer.populate_from_scene`` path uploads each piece's mesh and binds it
to its node. The room is open on the +Z side (no front wall) so the demo
camera can frame the interior from outside.

Geometry is intentionally minimal — simple boxes with per-mesh
``base_color`` tints. Add finer detail (book spines, plushies, posters with
real textures) once textured rendering lands.
"""
from __future__ import annotations

import numpy as np

from posecascade.assets.types import ImportedScene, Mesh
from posecascade.scene.component import MeshRefComponent
from posecascade.scene.node import Node
from posecascade.scene.scene import Scene
from posecascade.utils.math3d import vec3

Color = tuple[float, float, float, float]
Translation = tuple[float, float, float]
Size = tuple[float, float, float]

# Palette — warm wood floor, cream walls, pastel accents.
_FLOOR_COLOR: Color = (0.90, 0.78, 0.60, 1.0)
_WALL_COLOR: Color = (0.96, 0.92, 0.86, 1.0)
_CEILING_COLOR: Color = (0.98, 0.96, 0.94, 1.0)
_BED_FRAME_COLOR: Color = (0.45, 0.30, 0.20, 1.0)
_MATTRESS_COLOR: Color = (0.97, 0.94, 0.90, 1.0)
_PILLOW_COLOR: Color = (0.72, 0.86, 1.00, 1.0)
_BLANKET_COLOR: Color = (1.00, 0.72, 0.82, 1.0)
_DESK_COLOR: Color = (0.78, 0.58, 0.38, 1.0)
_DESK_LEG_COLOR: Color = (0.55, 0.40, 0.25, 1.0)
_MONITOR_BEZEL_COLOR: Color = (0.10, 0.10, 0.12, 1.0)
_MONITOR_SCREEN_COLOR: Color = (0.45, 0.70, 0.95, 1.0)
_WINDOW_COLOR: Color = (0.65, 0.88, 1.00, 1.0)
_DOOR_COLOR: Color = (0.55, 0.36, 0.22, 1.0)
_RUG_COLOR: Color = (0.95, 0.70, 0.85, 1.0)

# Default room dimensions (width X, height Y, depth Z) in metres.
_DEFAULT_SIZE: Size = (6.0, 3.0, 6.0)
_WALL_THICKNESS = 0.15


def make_anime_bedroom_scene(
    name: str = "anime_bedroom",
    size: Size = _DEFAULT_SIZE,
) -> ImportedScene:
    """Build a small bedroom: floor, 3 walls, ceiling, bed, desk, monitor, window, door, rug.

    Origin sits at the centre of the floor. The +Z side is intentionally open
    so the demo camera can see in.
    """
    builder = _BedroomBuilder(size=size)
    builder.add_room()
    builder.add_bed()
    builder.add_desk_with_monitor()
    builder.add_window()
    builder.add_door()
    builder.add_rug()
    return builder.finalize(name=name)


class _BedroomBuilder:
    """Accumulates pieces; each piece becomes a Mesh + Node bound by MeshRefComponent."""

    def __init__(self, size: Size) -> None:
        self._size: Size = size
        self._meshes: list[Mesh] = []
        self._nodes: list[Node] = []

    def add_room(self) -> None:
        width, height, depth = self._size
        wt = _WALL_THICKNESS
        # Floor: thin slab below y=0; top at y=0.
        self._add_box("floor", (width, wt, depth), (0.0, -wt * 0.5, 0.0), _FLOOR_COLOR)
        # Ceiling.
        self._add_box(
            "ceiling", (width, wt, depth), (0.0, height + wt * 0.5, 0.0), _CEILING_COLOR,
        )
        # Back wall (-Z).
        self._add_box(
            "wall_back", (width, height, wt),
            (0.0, height * 0.5, -depth * 0.5 - wt * 0.5), _WALL_COLOR,
        )
        # Left wall (-X).
        self._add_box(
            "wall_left", (wt, height, depth),
            (-width * 0.5 - wt * 0.5, height * 0.5, 0.0), _WALL_COLOR,
        )
        # Right wall (+X).
        self._add_box(
            "wall_right", (wt, height, depth),
            (width * 0.5 + wt * 0.5, height * 0.5, 0.0), _WALL_COLOR,
        )

    def add_bed(self) -> None:
        """Bed in the back-right corner, headboard against the back wall."""
        bed_x = self._size[0] * 0.5 - 1.0
        bed_z = -self._size[2] * 0.5 + 1.1
        # Frame — large flat box.
        frame_size = (1.8, 0.35, 2.1)
        self._add_box("bed_frame", frame_size, (bed_x, frame_size[1] * 0.5, bed_z),
                      _BED_FRAME_COLOR)
        # Mattress on top.
        mat_size = (1.7, 0.18, 2.0)
        mat_y = frame_size[1] + mat_size[1] * 0.5
        self._add_box("bed_mattress", mat_size, (bed_x, mat_y, bed_z), _MATTRESS_COLOR)
        # Pillow at headboard end (-Z relative to mattress).
        pillow_size = (1.4, 0.10, 0.45)
        pillow_y = mat_y + mat_size[1] * 0.5 + pillow_size[1] * 0.5
        pillow_z = bed_z - mat_size[2] * 0.5 + pillow_size[2] * 0.5 + 0.05
        self._add_box("bed_pillow", pillow_size, (bed_x, pillow_y, pillow_z), _PILLOW_COLOR)
        # Blanket covering the lower 2/3 of the mattress.
        blanket_size = (1.65, 0.05, 1.35)
        blanket_y = mat_y + mat_size[1] * 0.5 + blanket_size[1] * 0.5
        blanket_z = bed_z + 0.30
        self._add_box(
            "bed_blanket", blanket_size, (bed_x, blanket_y, blanket_z), _BLANKET_COLOR,
        )

    def add_desk_with_monitor(self) -> None:
        """Desk against the left wall, monitor on top."""
        desk_x = -self._size[0] * 0.5 + 0.85
        desk_z = 0.5
        top_size = (1.4, 0.06, 0.7)
        top_y = 0.75
        self._add_box(
            "desk_top", top_size, (desk_x, top_y, desk_z), _DESK_COLOR,
        )
        # Four legs (front-left, front-right, back-left, back-right relative to desk).
        leg_size = (0.06, top_y - top_size[1], 0.06)
        leg_y = (top_y - top_size[1]) * 0.5
        for sx in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                offset_x = sx * (top_size[0] * 0.5 - 0.08)
                offset_z = sz * (top_size[2] * 0.5 - 0.08)
                self._add_box(
                    f"desk_leg_{int(sx)}_{int(sz)}", leg_size,
                    (desk_x + offset_x, leg_y, desk_z + offset_z),
                    _DESK_LEG_COLOR,
                )
        # Monitor on top.
        bezel_size = (0.65, 0.40, 0.04)
        bezel_y = top_y + top_size[1] * 0.5 + bezel_size[1] * 0.5 + 0.05
        bezel_z = desk_z - 0.22
        self._add_box(
            "monitor_bezel", bezel_size, (desk_x, bezel_y, bezel_z), _MONITOR_BEZEL_COLOR,
        )
        # Glowing-blue screen — slightly forward, smaller than bezel.
        screen_size = (0.58, 0.34, 0.005)
        self._add_box(
            "monitor_screen", screen_size,
            (desk_x, bezel_y, bezel_z + bezel_size[2] * 0.5 + 0.003),
            _MONITOR_SCREEN_COLOR,
        )
        # Monitor stand: a small block under the bezel.
        stand_size = (0.12, 0.10, 0.18)
        stand_y = top_y + top_size[1] * 0.5 + stand_size[1] * 0.5
        self._add_box(
            "monitor_stand", stand_size, (desk_x, stand_y, bezel_z), _MONITOR_BEZEL_COLOR,
        )

    def add_window(self) -> None:
        """Sky-blue panel set into the back wall."""
        win_size = (1.8, 1.2, 0.02)
        win_y = self._size[1] * 0.55
        win_z = -self._size[2] * 0.5 + 0.01  # in front of the back wall
        self._add_box(
            "window", win_size, (-1.0, win_y, win_z), _WINDOW_COLOR,
        )

    def add_door(self) -> None:
        """Wooden door panel set into the right wall."""
        door_size = (0.02, 2.0, 0.9)
        door_x = self._size[0] * 0.5 - 0.01  # in front of the right wall
        self._add_box(
            "door", door_size, (door_x, door_size[1] * 0.5, 1.5), _DOOR_COLOR,
        )

    def add_rug(self) -> None:
        """Pink rug covering the centre of the room."""
        rug_size = (3.0, 0.02, 2.4)
        self._add_box(
            "rug", rug_size, (0.0, rug_size[1] * 0.5, 0.5), _RUG_COLOR,
        )

    def finalize(self, name: str) -> ImportedScene:
        scene = Scene(name=name)
        for node in self._nodes:
            scene.root.add_child(node)
        return ImportedScene(meshes=tuple(self._meshes), scene=scene)

    def _add_box(
        self, piece_name: str, size: Size, translation: Translation, color: Color,
    ) -> None:
        mesh = _make_box_mesh(piece_name, size, color)
        node = Node(name=piece_name)
        node.transform.set_translation(vec3(*translation))
        node.add_component(MeshRefComponent(mesh_indices=(len(self._meshes),)))
        self._meshes.append(mesh)
        self._nodes.append(node)


def _make_box_mesh(name: str, size: Size, color: Color) -> Mesh:
    """Axis-aligned box centred on origin with outward-facing per-vertex normals."""
    half_x, half_y, half_z = size[0] * 0.5, size[1] * 0.5, size[2] * 0.5
    # 6 faces × 4 corners = 24 vertices, 6 × 6 = 36 indices, normals match face direction.
    faces = (
        # (normal, v0, v1, v2, v3) — CCW order looking along the inverse normal
        ((1, 0, 0), ( half_x, -half_y,  half_z), ( half_x,  half_y,  half_z),
         ( half_x,  half_y, -half_z), ( half_x, -half_y, -half_z)),
        ((-1, 0, 0), (-half_x, -half_y, -half_z), (-half_x,  half_y, -half_z),
         (-half_x,  half_y,  half_z), (-half_x, -half_y,  half_z)),
        ((0, 1, 0), (-half_x,  half_y, -half_z), ( half_x,  half_y, -half_z),
         ( half_x,  half_y,  half_z), (-half_x,  half_y,  half_z)),
        ((0, -1, 0), (-half_x, -half_y,  half_z), ( half_x, -half_y,  half_z),
         ( half_x, -half_y, -half_z), (-half_x, -half_y, -half_z)),
        ((0, 0, 1), (-half_x, -half_y,  half_z), ( half_x, -half_y,  half_z),
         ( half_x,  half_y,  half_z), (-half_x,  half_y,  half_z)),
        ((0, 0, -1), ( half_x, -half_y, -half_z), (-half_x, -half_y, -half_z),
         (-half_x,  half_y, -half_z), ( half_x,  half_y, -half_z)),
    )
    positions = np.empty((24, 3), dtype=np.float32)
    normals = np.empty((24, 3), dtype=np.float32)
    indices = np.empty(36, dtype=np.uint32)
    for face_index, (normal, v0, v1, v2, v3) in enumerate(faces):
        base = face_index * 4
        positions[base + 0] = v0
        positions[base + 1] = v1
        positions[base + 2] = v2
        positions[base + 3] = v3
        normals[base:base + 4] = normal
        idx_base = face_index * 6
        indices[idx_base:idx_base + 6] = (base, base + 1, base + 2, base, base + 2, base + 3)
    return Mesh(
        name=name, positions=positions, indices=indices, normals=normals,
        base_color=color,
    )
