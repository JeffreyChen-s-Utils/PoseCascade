"""Render-time numeric constants. Magic numbers in render code MUST live here."""
from __future__ import annotations

MAX_BONES_PER_VERTEX = 4
MAX_LIGHTS_FORWARD = 8
SHADOW_MAP_SIZE = 2048
DEFAULT_NEAR_PLANE = 0.05
DEFAULT_FAR_PLANE = 1000.0
DEFAULT_FOV_DEGREES = 60.0

# Asset hard caps — refuse files exceeding these.
MAX_MESH_VERTICES = 8_000_000
MAX_MESH_INDICES = 32_000_000
MAX_TEXTURE_DIMENSION = 8192
MAX_EMBEDDED_BUFFER_BYTES = 256 * 1024 * 1024

# PMX/PMD specific caps. PMX supports 4-byte indices so the spec ceiling is
# 2^31, but real-world models stay well under the values below; refusing
# anything bigger is a cheap defence against malformed or hostile files.
MAX_PMX_VERTEX_COUNT = 2_000_000
MAX_PMX_FACE_COUNT = 6_000_000
MAX_PMX_BONE_COUNT = 4_096
MAX_PMX_MORPH_COUNT = 4_096
MAX_PMX_MATERIAL_COUNT = 1_024
MAX_PMX_TEXTURE_COUNT = 1_024
MAX_PMX_RIGID_BODY_COUNT = 4_096
MAX_PMX_JOINT_COUNT = 4_096
MAX_PMX_DISPLAY_FRAME_COUNT = 256
MAX_PMX_TEXT_BYTES = 1 << 16    # per-string cap for length-prefixed PMX strings
