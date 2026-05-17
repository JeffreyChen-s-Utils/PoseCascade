"""Forward renderer: walks a :class:`~posecascade.scene.scene.Scene` and draws it.

The renderer owns the basic forward shader program and the GPU mesh cache.
``draw`` is meant to be called from :meth:`~posecascade.ui.viewport.Viewport.paintGL`
on the GL-owning thread; cross-thread calls trip
:meth:`~posecascade.gl.context.GLContext.assert_owned`.
"""
from __future__ import annotations

import ctypes
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_BACK,
    GL_BLEND,
    GL_CLAMP_TO_BORDER,
    GL_COLOR_BUFFER_BIT,
    GL_CULL_FACE,
    GL_DEPTH_ATTACHMENT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_COMPONENT,
    GL_DEPTH_COMPONENT24,
    GL_DEPTH_TEST,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_BINDING,
    GL_FRAMEBUFFER_SRGB,
    GL_FRONT,
    GL_NEAREST,
    GL_NONE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_SRC_ALPHA,
    GL_STATIC_DRAW,
    GL_TEXTURE0,
    GL_TEXTURE1,
    GL_TEXTURE2,
    GL_TEXTURE_2D,
    GL_TEXTURE_BORDER_COLOR,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRIANGLES,
    GL_TRUE,
    GL_UNSIGNED_INT,
    glActiveTexture,
    glBindBuffer,
    glBindFramebuffer,
    glBindTexture,
    glBindVertexArray,
    glBlendFunc,
    glBufferData,
    glClear,
    glClearColor,
    glCullFace,
    glDepthMask,
    glDisable,
    glDrawArrays,
    glDrawBuffer,
    glDrawElements,
    glEnable,
    glEnableVertexAttribArray,
    glFramebufferTexture2D,
    glGenBuffers,
    glGenFramebuffers,
    glGenTextures,
    glGenVertexArrays,
    glGetIntegerv,
    glReadBuffer,
    glTexImage2D,
    glTexParameterfv,
    glTexParameteri,
    glUniform1f,
    glUniform1i,
    glUniform3fv,
    glUniform4fv,
    glUniformMatrix3fv,
    glUniformMatrix4fv,
    glUseProgram,
    glVertexAttribPointer,
    glViewport,
)

from posecascade.assets.types import ImportedScene, Mesh, Skin
from posecascade.errors import GLError
from posecascade.gl.binding import bind_vao, use_program
from posecascade.gl.compute_skin import PassiveSkinDispatcher, build_exclude_bits
from posecascade.gl.mesh_uploader import (
    GLMesh,
    reupload_normal_vbo,
    reupload_position_vbo,
    reupload_texcoords_vbo,
    upload_mesh,
)
from posecascade.gl.shader import Program, compile_program
from posecascade.gl.texture import (
    GLTexture,
    make_white_fallback,
    set_toon_sampler_params,
    upload_texture,
)
from posecascade.gl.uniforms import (
    U_AMBIENT,
    U_BASE_COLOR,
    U_BASE_COLOR_TEX,
    U_BONE_DQ_DUAL,
    U_BONE_DQ_REAL,
    U_BONE_MATRICES,
    U_CELL_SIZE,
    U_COLOR_A,
    U_COLOR_B,
    U_EDGE_COLOR,
    U_EDGE_SIZE,
    U_FADE_END,
    U_FADE_START,
    U_GROUND_COLOR,
    U_GROUND_Y,
    U_HORIZON_COLOR,
    U_HORIZON_Y,
    U_LIGHT_COLOR,
    U_LIGHT_DIRECTION,
    U_LIGHT_SPACE_MATRIX,
    U_MODEL_MATRIX,
    U_NORMAL_MATRIX,
    U_PROJECTION_MATRIX,
    U_SECONDARY_LIGHT_COLORS,
    U_SECONDARY_LIGHT_COUNT,
    U_SECONDARY_LIGHT_DIRECTIONS,
    U_SHADOW_COLOR,
    U_SHADOW_ENABLED,
    U_SHADOW_MAP,
    U_SHADOW_STRENGTH,
    U_SPECULAR,
    U_SPECULAR_POWER,
    U_SPHERE_MODE,
    U_SPHERE_TEX,
    U_TOON_TEX,
    U_VIEW_MATRIX,
    U_ZENITH_COLOR,
)
from posecascade.render.camera import Camera
from posecascade.render.effects.chain import EffectChain
from posecascade.render.effects.executor import EffectChainExecutor
from posecascade.render.effects.ping_pong import EffectPingPong
from posecascade.render.material import MMDMaterial
from posecascade.render.toon_promote import (
    default_toon_material,
    default_toon_ramp_pixels,
)
from posecascade.scene.component import ClothComponent, MeshRefComponent, SkinRefComponent
from posecascade.scene.node import Node
from posecascade.scene.scene import Scene
from posecascade.utils.dual_quaternion import matrices_to_dual_quaternions
from posecascade.utils.logging import get_logger
from posecascade.utils.math3d import Mat4
from posecascade.utils.profiling import frame_section

_logger = get_logger(__name__)
_warned_oversize_skins: set[int] = set()

_DEFAULT_BASE_COLOR = (0.8, 0.8, 0.85, 1.0)
_MAX_BONES = 384  # must match skinned.vert's MAX_BONES define
_DEFAULT_LIGHT_DIRECTION = (0.3, 0.7, 0.6)
_DEFAULT_LIGHT_COLOR = (1.0, 1.0, 1.0)
# Selection overlay: a second inverted-hull outline pass run on every mesh
# under the selected holder using a thicker edge and a bright contrast
# colour so the user can see which model is selected regardless of whether
# its materials enabled edges. The edge size is in mesh-local units —
# typical MMD models live around the unit scale, so 0.04 reads cleanly at
# normal camera distance without dominating the silhouette.
_SELECTION_EDGE_SIZE = 0.04
_SELECTION_EDGE_COLOR = (1.0, 0.85, 0.10, 1.0)
# Procedural MMD-style checkered ground. The plane is a single quad that
# covers ±_GROUND_HALF_EXTENT metres; the fragment shader does the
# checker pattern based on world-space XZ + a radial fade so the floor
# blends to the clear colour near the horizon instead of cutting off
# at a hard edge.
_GROUND_HALF_EXTENT = 25.0
_GROUND_Y_LEVEL = 0.0
_GROUND_CELL_SIZE = 0.5
_GROUND_COLOR_A = (0.78, 0.79, 0.82, 1.0)
_GROUND_COLOR_B = (0.62, 0.64, 0.68, 1.0)
_GROUND_FADE_START = 6.0
_GROUND_FADE_END = 16.0
# Projected ground shadow. Drawn after the ground with depth write off
# and the body's silhouette flattened onto the ground plane in the
# vertex shader. Alpha < 1 so the checker still reads through.
_PROJECTED_SHADOW_COLOR = (0.0, 0.0, 0.0, 0.35)
# Depth-map self-shadow. The scene is rendered once from the light's
# point of view into a depth texture, then the toon shader compares
# each fragment's light-space depth against it. 1024² is enough to
# resolve a typical MMD character at editor zoom without obvious
# stair-stepping; bump for larger scenes if banding appears. The
# orthographic frustum is sized to cover a ``±_SHADOW_HALF_EXTENT``
# cube centred on the world origin — large enough for a single dance
# stage, conservative enough that the depth precision stays usable.
_SHADOW_MAP_SIZE = 1024
_SHADOW_HALF_EXTENT = 4.0
_SHADOW_NEAR = 0.1
_SHADOW_FAR = 20.0
_SHADOW_LIGHT_DISTANCE = 10.0
# How much darker fragments in shadow become vs full daylight. 0 turns
# the shadow off, 1 makes shadowed areas pitch black. 0.45 matches the
# default Standard shader in MMD reasonably well — clearly visible but
# never crushes the under-chin / inner-skirt areas to flat black.
_SHADOW_STRENGTH = 0.45
# Texture unit reserved for the shadow map sampler in the toon pass.
# Units 0..2 are already taken by base-colour / sphere / toon ramp.
_TEX_UNIT_SHADOW = 3
# Gradient skybox: warm horizon to cool zenith, with a darker ground
# tint below the horizon line. Values are sRGB-encoded so they land on
# screen unchanged after ``GL_FRAMEBUFFER_SRGB`` re-encodes them — the
# colours were picked against a calibrated sRGB display.
_SKY_ZENITH_COLOR = (0.34, 0.45, 0.62)
_SKY_HORIZON_COLOR = (0.78, 0.74, 0.66)
_SKY_GROUND_COLOR = (0.18, 0.18, 0.20)
_SKY_HORIZON_Y = 0.42
# Multi-light HighDef support. The toon shader accepts up to three
# secondary directional lights on top of the primary; the engine
# clamps caller input to this constant. Match the GLSL ``MAX_SECONDARY_
# LIGHTS`` define in shaders/toon/toon.frag.
_MAX_SECONDARY_LIGHTS = 3
# Built-in light preset for the "HighDef" look — a soft above-back rim
# + a cool below-front fill. Tunable via :meth:`Renderer.set_secondary_lights`.
_HIGHDEF_RIM_LIGHT = ((-0.5, 0.7, -0.7), (0.42, 0.50, 0.60))
_HIGHDEF_FILL_LIGHT = ((0.0, -0.3, 0.95), (0.18, 0.20, 0.24))
# Lower bound on vector magnitude before we treat it as degenerate —
# applied across the light-direction normalisation, the look-at basis,
# and the perspective-divide guard in the shadow projection. 1e-6 sits
# comfortably above float32 ULP at unit scale while still catching
# accidental zero vectors.
_DEGENERATE_VEC_EPS = 1.0e-6
# Maximum |dot(light_dir, up)| before we swap the look-at's ``up`` to
# +Z to avoid a singular cross product. 0.999 = ~2.5° — generous,
# never triggers for typical light directions.
_LOOK_AT_PARALLEL_EPS = 0.999


@dataclass
class _ProgramFrameState:
    """Per-program record of which uniform groups have been uploaded this frame.

    GL keeps uniform values per-program until they're overwritten, so we
    only need to upload the camera / lights / shadow / bone-matrix
    blocks once per program per frame even when dozens of meshes draw
    through that program. Cleared at the top of every
    :meth:`Renderer.draw`.
    """

    camera_uploaded: bool = False
    lights_uploaded: bool = False
    shadow_uploaded: bool = False
    skin_uploaded: set[int] = field(default_factory=set)


@dataclass
class Renderer:
    """Owns the shader programs and the mesh-id → :class:`GLMesh` cache."""

    shaders_root: Path
    base_color: tuple[float, float, float, float] = _DEFAULT_BASE_COLOR
    _program: Program | None = field(default=None, init=False)
    _skin_program: Program | None = field(default=None, init=False)
    _toon_program: Program | None = field(default=None, init=False)
    _toon_skin_program: Program | None = field(default=None, init=False)
    # Dual-quaternion skinning variant of the toon-skinned program. Compiled
    # alongside the LBS one; selected at draw time when ``_dqs_enabled``
    # is true. Defaults ON — LBS visibly pinches wrists / elbows / shoulders
    # under the deep joint bends common in pose-driven content (e.g. the
    # dog_crawl example: 90°+ elbow + wrist twist made fingers collapse to
    # a spike). DQS keeps joint volume there. The CPU cost is the
    # matrix→dual-quaternion conversion in :meth:`_upload_skin_uniforms`,
    # ~tens of µs per skin per frame; well under our budget.
    _toon_skin_dqs_program: Program | None = field(default=None, init=False)
    _dqs_enabled: bool = field(default=True, init=False)
    # Force-toon-shading: when on, meshes without an ``MMDMaterial`` (glTF,
    # OBJ, FBX, …) get routed through the toon pipeline using
    # :func:`default_toon_material` + a procedural 2-band ramp. Off by
    # default so non-MMD imports keep their native PBR / forward look —
    # turn on for "MMD-ify any model" scenes.
    _force_toon_shading: bool = field(default=False, init=False)
    _default_toon_material: MMDMaterial | None = field(default=None, init=False)
    _default_toon_texture_id: int = field(default=0, init=False)
    _outline_program: Program | None = field(default=None, init=False)
    _outline_skin_program: Program | None = field(default=None, init=False)
    _meshes: dict[int, GLMesh] = field(default_factory=dict, init=False)
    _node_to_mesh: dict[int, list[int]] = field(default_factory=dict, init=False)
    _node_to_skin: dict[int, Skin] = field(default_factory=dict, init=False)
    # Per-frame skin → bone matrix cache. The same skin is referenced
    # by multiple draws each frame (toon main pass + outline pass +
    # shadow pass + ground-projected shadow), and ``_compute_bone_matrices``
    # walks 354 bone parent chains every call. Cleared at the start of
    # each ``draw()`` and populated lazily on first request — turns
    # ~10 walks per frame on the bundled rig into 1.
    _frame_bone_matrix_cache: dict[int, np.ndarray] = field(
        default_factory=dict, init=False,
    )
    # Lookup table populated alongside ``_node_to_mesh``: maps the importer-side
    # ``(node_object_id, imported_mesh_index)`` pair to the renderer-internal mesh id.
    # Cloth and other dynamic-mesh systems use this to find the GPU buffers they
    # need to refresh each frame from a logical mesh reference.
    _node_mesh_index_to_id: dict[tuple[int, int], int] = field(default_factory=dict, init=False)
    # Reverse of the importer's flat-mesh index — built at populate time so the
    # morph applier can stream new vertex data into every Mesh derived from a
    # PMX material, and the material-override map can address each Mesh by its
    # importer-side index.
    _mesh_index_to_id: dict[int, int] = field(default_factory=dict, init=False)
    _id_to_mesh_index: dict[int, int] = field(default_factory=dict, init=False)
    _textures: list[GLTexture] = field(default_factory=list, init=False)
    _white_fallback: GLTexture | None = field(default=None, init=False)
    # Live light + self-shadow state. The toon pass reads these every
    # draw — they replace the previous compile-time constants in the
    # shader so VMD-driven lighting can flow into the visible output.
    _light_direction: tuple[float, float, float] = field(
        default=_DEFAULT_LIGHT_DIRECTION, init=False,
    )
    _light_color: tuple[float, float, float] = field(
        default=_DEFAULT_LIGHT_COLOR, init=False,
    )
    # Secondary directional lights. Up to ``_MAX_SECONDARY_LIGHTS`` —
    # each entry is ``(direction, color)`` with direction pointing
    # toward the source. Empty tuple = HighDef off; the toon pass
    # behaves exactly as a single-light setup.
    _secondary_lights: tuple[
        tuple[tuple[float, float, float], tuple[float, float, float]], ...
    ] = field(default=(), init=False)
    # Active material overrides keyed on the same flat-mesh index. ``apply``
    # of a MorphSnapshot replaces this entire dict, so a morph that fades
    # back to zero weight automatically stops shadowing its base material.
    _material_overrides: dict[int, MMDMaterial] = field(default_factory=dict, init=False)
    # Lazily allocated post-effect ping-pong; only instantiated once the
    # caller requests :meth:`apply_effect_chain`.
    _effect_ping_pong: EffectPingPong | None = field(default=None, init=False)
    # Currently selected top-level holder. ``draw`` runs an extra outline
    # pass on every mesh under this node so the user can see which model
    # they picked. ``None`` skips the selection overlay entirely.
    _selected_holder: Node | None = field(default=None, init=False)
    # Checker-ground + projected-shadow pass state. Programs are compiled
    # lazily in :meth:`initialize`; the quad VAO is built once at init
    # time and reused every frame. ``_ground_enabled`` defaults to True
    # so a fresh editor session shows the floor without an explicit
    # toggle; callers can opt out per-project via :meth:`set_ground_enabled`.
    _ground_program: Program | None = field(default=None, init=False)
    _shadow_proj_program: Program | None = field(default=None, init=False)
    _shadow_proj_skin_program: Program | None = field(default=None, init=False)
    _ground_vao: int = field(default=0, init=False)
    _ground_vbo: int = field(default=0, init=False)
    _ground_ebo: int = field(default=0, init=False)
    _ground_index_count: int = field(default=0, init=False)
    _ground_enabled: bool = field(default=True, init=False)
    _projected_shadow_enabled: bool = field(default=True, init=False)
    # Depth-map self-shadow programs + FBO. The depth pass renders all
    # opaque scene meshes from the light's POV into ``_shadow_depth_tex``
    # before the main forward pass starts; the toon program then samples
    # that texture per-fragment for self-occlusion.
    _shadow_depth_program: Program | None = field(default=None, init=False)
    _shadow_depth_skin_program: Program | None = field(default=None, init=False)
    _shadow_fbo: int = field(default=0, init=False)
    _shadow_depth_tex: int = field(default=0, init=False)
    _self_shadow_enabled: bool = field(default=True, init=False)
    # sRGB-aware output: when on, ``GL_FRAMEBUFFER_SRGB`` is enabled for
    # the duration of ``draw`` and the GPU re-encodes shader output to
    # sRGB on write. Combined with sRGB-decoded base-colour textures
    # this lets every shader work in linear space, matching how modern
    # engines render. Default ON — the few golden-image tests pinned to
    # the old linear baseline opt out with ``set_srgb_output_enabled``.
    _srgb_output_enabled: bool = field(default=True, init=False)
    # Sky pass state. The vertex shader synthesises a fullscreen
    # triangle from ``gl_VertexID``, so the renderer only needs an
    # empty VAO bound at draw time.
    _sky_program: Program | None = field(default=None, init=False)
    _sky_vao: int = field(default=0, init=False)
    _sky_enabled: bool = field(default=True, init=False)
    # Cached light-space matrix and its float32 form. Recomputed once at
    # the start of each draw frame from the current light direction so
    # the depth pass and the toon pass see the same transform.
    _light_space_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float32), init=False,
    )
    # Per-frame float32 caches. ``_set_camera_uniforms`` and friends
    # used to call ``np.ascontiguousarray(view, dtype=np.float32)`` on
    # every mesh of every pass, copying the same 4×4 matrix into a new
    # buffer ~90 times / frame. The draw loop now converts once at the
    # top into these slots and the helpers read them directly. ``None``
    # outside of an active :meth:`draw` so a stale value can't leak
    # between frames.
    _view_f32: np.ndarray | None = field(default=None, init=False)
    _proj_f32: np.ndarray | None = field(default=None, init=False)
    _light_direction_f32: np.ndarray | None = field(default=None, init=False)
    _light_color_f32: np.ndarray | None = field(default=None, init=False)
    _light_space_matrix_f32: np.ndarray | None = field(default=None, init=False)
    _secondary_directions_f32: np.ndarray | None = field(default=None, init=False)
    _secondary_colors_f32: np.ndarray | None = field(default=None, init=False)
    # GPU compute-shader dispatcher for passive-skin-deform cloth pieces.
    # ``None`` when the active context lacks GL 4.3 / compute support —
    # the renderer transparently falls back to the host's CPU LBS path.
    _passive_skin_dispatcher: PassiveSkinDispatcher | None = field(
        default=None, init=False,
    )
    # Per-frame, per-program uniform-state cache. GL uniform state is
    # persistent across draws within a program — once a uniform is set,
    # it stays until overwritten. The previous code re-uploaded the same
    # view / proj / light / shadow / bone-matrix uniforms on every mesh
    # of every pass, costing ~3 ms / frame on the Herta scene's
    # ~30 meshes × 3 passes × 8 redundant uniforms each. Track here
    # which uniform groups have already been written to each program
    # this frame so subsequent draws skip the ascontiguousarray +
    # glUniform* roundtrip. Cleared at the start of every ``draw()``.
    _program_frame_state: dict[int, _ProgramFrameState] = field(
        default_factory=dict, init=False,
    )

    def initialize(self) -> None:
        """Compile shaders. Must run on the GL-owning thread after the context is current."""
        forward = self.shaders_root / "forward"
        basic_vert = (forward / "basic.vert").read_text(encoding="utf-8")
        skinned_vert = (forward / "skinned.vert").read_text(encoding="utf-8")
        frag = (forward / "basic.frag").read_text(encoding="utf-8")
        self._program = compile_program(basic_vert, frag)
        self._skin_program = compile_program(skinned_vert, frag)
        toon = self.shaders_root / "toon"
        if toon.is_dir():
            toon_vert = (toon / "toon.vert").read_text(encoding="utf-8")
            toon_skinned_vert = (toon / "toon_skinned.vert").read_text(encoding="utf-8")
            toon_frag = (toon / "toon.frag").read_text(encoding="utf-8")
            outline_vert = (toon / "outline.vert").read_text(encoding="utf-8")
            outline_skinned_vert = (toon / "outline_skinned.vert").read_text(encoding="utf-8")
            outline_frag = (toon / "outline.frag").read_text(encoding="utf-8")
            self._toon_program = compile_program(toon_vert, toon_frag)
            self._toon_skin_program = compile_program(toon_skinned_vert, toon_frag)
            self._outline_program = compile_program(outline_vert, outline_frag)
            self._outline_skin_program = compile_program(outline_skinned_vert, outline_frag)
            dqs_path = toon / "toon_skinned_dqs.vert"
            if dqs_path.is_file():
                toon_dqs_vert = dqs_path.read_text(encoding="utf-8")
                self._toon_skin_dqs_program = compile_program(toon_dqs_vert, toon_frag)
        ground = self.shaders_root / "ground"
        if ground.is_dir():
            ground_vert = (ground / "ground.vert").read_text(encoding="utf-8")
            ground_frag = (ground / "ground.frag").read_text(encoding="utf-8")
            shadow_vert = (ground / "shadow_projection.vert").read_text(encoding="utf-8")
            shadow_skin_vert = (ground / "shadow_projection_skinned.vert").read_text(
                encoding="utf-8",
            )
            shadow_frag = (ground / "shadow_projection.frag").read_text(encoding="utf-8")
            self._ground_program = compile_program(ground_vert, ground_frag)
            self._shadow_proj_program = compile_program(shadow_vert, shadow_frag)
            self._shadow_proj_skin_program = compile_program(shadow_skin_vert, shadow_frag)
            self._build_ground_geometry()
        self._compile_shadow_programs(self.shaders_root / "shadow")
        self._compile_sky_program(self.shaders_root / "sky")
        glEnable(GL_DEPTH_TEST)
        self._white_fallback = make_white_fallback()
        self._build_default_toon_assets()
        # Compute-shader path for passive-skin-deform cloth. Optional —
        # returns None on macOS legacy GL or any context that fails to
        # compile the compute stage. Callers stay on the CPU LBS path
        # in that case; no other rendering toggle is affected.
        passive_shader = self.shaders_root / "passive_skin" / "passive_skin_push.comp"
        if passive_shader.is_file():
            self._passive_skin_dispatcher = PassiveSkinDispatcher.try_create(
                passive_shader,
            )

    def _compile_shadow_programs(self, shadow_root: Path) -> None:
        """Compile the depth-only programs + build the shadow FBO if the dir exists."""
        if not shadow_root.is_dir():
            return
        depth_vert = (shadow_root / "depth_only.vert").read_text(encoding="utf-8")
        depth_skin_vert = (shadow_root / "depth_only_skinned.vert").read_text(
            encoding="utf-8",
        )
        depth_frag = (shadow_root / "depth_only.frag").read_text(encoding="utf-8")
        self._shadow_depth_program = compile_program(depth_vert, depth_frag)
        self._shadow_depth_skin_program = compile_program(depth_skin_vert, depth_frag)
        self._build_shadow_fbo()

    def _compile_sky_program(self, sky_root: Path) -> None:
        """Compile the gradient-sky program + allocate its empty VAO."""
        if not sky_root.is_dir():
            return
        sky_vert = (sky_root / "gradient.vert").read_text(encoding="utf-8")
        sky_frag = (sky_root / "gradient.frag").read_text(encoding="utf-8")
        self._sky_program = compile_program(sky_vert, sky_frag)
        self._sky_vao = int(glGenVertexArrays(1))

    def _build_default_toon_assets(self) -> None:
        """Build the procedural ramp + synthetic material used by force-toon.

        Factored out of :meth:`initialize` to keep that method under the
        ``PLR0915`` cognitive-complexity limit. Linear upload (NOT
        sRGB) — toon ramps encode a Lambert LUT, not display-referred
        colour. ``set_toon_sampler_params`` then switches to ``NEAREST``
        + ``CLAMP_TO_EDGE`` so the cel band is crisp.
        """
        self._default_toon_material = default_toon_material()
        default_ramp = upload_texture(default_toon_ramp_pixels(), srgb=False)
        set_toon_sampler_params(default_ramp.texture_id)
        self._textures.append(default_ramp)
        self._default_toon_texture_id = default_ramp.texture_id

    def _build_shadow_fbo(self) -> None:
        """Allocate the depth-only FBO + texture used for self-shadow sampling.

        Uses ``GL_DEPTH_COMPONENT24`` with ``CLAMP_TO_BORDER`` + a white
        border so fragments outside the light's frustum sample depth = 1
        (max far plane, never in shadow). Filter is ``GL_NEAREST`` — PCF
        is a future polish item, but the unfiltered comparison already
        looks fine for the MMD-style hard self-shadow.

        ``initialize`` may run while the caller already has an offscreen
        FBO bound (e.g. the offscreen-render test fixtures); we save and
        restore that binding so configuring the shadow target doesn't
        leak state outward.
        """
        previous_fbo = int(glGetIntegerv(GL_FRAMEBUFFER_BINDING))
        depth_tex = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, depth_tex)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24,
            _SHADOW_MAP_SIZE, _SHADOW_MAP_SIZE, 0,
            GL_DEPTH_COMPONENT, GL_FLOAT, None,
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        border = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border)
        glBindTexture(GL_TEXTURE_2D, 0)

        fbo = int(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_tex, 0,
        )
        # Depth-only — no colour attachment. The OpenGL spec requires us
        # to explicitly say "no draw buffer" or the FBO is incomplete.
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)
        glBindFramebuffer(GL_FRAMEBUFFER, previous_fbo)

        self._shadow_fbo = fbo
        self._shadow_depth_tex = depth_tex

    def _build_ground_geometry(self) -> None:
        """Upload a single-quad ``±_GROUND_HALF_EXTENT`` XZ plane at ``y=0``.

        The checker pattern is computed in the fragment shader from world
        coordinates, so a single 4-vertex quad is enough; we don't need
        tessellation for the visual effect.
        """
        half = _GROUND_HALF_EXTENT
        positions = np.array(
            [
                [-half, _GROUND_Y_LEVEL, -half],
                [ half, _GROUND_Y_LEVEL, -half],
                [ half, _GROUND_Y_LEVEL,  half],
                [-half, _GROUND_Y_LEVEL,  half],
            ],
            dtype=np.float32,
        )
        indices = np.array([0, 2, 1, 0, 3, 2], dtype=np.uint32)
        vao = int(glGenVertexArrays(1))
        glBindVertexArray(vao)
        vbo = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        ebo = int(glGenBuffers(1))
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        glBindVertexArray(0)
        self._ground_vao = vao
        self._ground_vbo = vbo
        self._ground_ebo = ebo
        self._ground_index_count = int(indices.size)

    def attach_mesh(self, node: Node, mesh: Mesh) -> int:
        """Upload ``mesh`` to GPU and bind it to ``node`` for drawing.

        Convenience for procedural / single-mesh nodes; importers should use
        :meth:`populate_from_scene` so every node's :class:`MeshRefComponent`
        is honoured.
        """
        mesh_id = len(self._meshes)
        self._meshes[mesh_id] = upload_mesh(mesh)
        self._node_to_mesh.setdefault(id(node), []).append(mesh_id)
        return mesh_id

    def populate_from_slots(self, slots) -> Scene:    # noqa: ANN001 — late-bound SceneSlots
        """Compose every visible slot into a unified scene + register meshes.

        Returns a freshly built composite :class:`Scene` whose root holds
        one synthetic transform Node per slot (carrying that slot's
        ``transform``); the slot's imported children are re-parented
        under it so the draw-time ``_world_matrix`` walk naturally folds
        the slot offset into every bone Node's world transform.

        The mutation is one-way — the imported scene's original root
        ends up empty after this call. Re-loading a slot requires the
        caller to re-import the model rather than calling this twice on
        the same :class:`SceneSlots`.
        """
        composite = Scene(name="slots_root")
        for slot in slots:
            if slot.imported.scene is None or not slot.visible:
                continue
            self.populate_from_scene(slot.imported)
            slot_root = Node(name=f"slot_{slot.name}", transform=slot.transform)
            for child in list(slot.imported.scene.root.children):
                slot.imported.scene.root.remove_child(child)
                slot_root.add_child(child)
            composite.root.add_child(slot_root)
        return composite

    def populate_from_scene(self, imported: ImportedScene) -> None:
        """Upload every mesh + texture referenced by the imported scene.

        Walks the imported scene; for every node with a ``MeshRefComponent``,
        uploads each referenced mesh exactly once and links any base-colour
        texture from ``imported.textures``. Nodes that also carry a
        :class:`SkinRefComponent` are flagged for skinning so the renderer
        switches to the skin shader and uploads bone matrices per draw.
        """
        if imported.scene is None:
            return
        flat_meshes = imported.meshes
        image_to_gl_tex = self._upload_textures(imported)
        uploaded: dict[int, int] = {}
        for node in imported.scene.root.traverse():
            skin = _find_skin(node)
            if skin is not None:
                self._node_to_skin[id(node)] = skin
            for component in node.components:
                if not isinstance(component, MeshRefComponent):
                    continue
                for mesh_index in component.mesh_indices:
                    if mesh_index < 0 or mesh_index >= len(flat_meshes):
                        continue
                    cached = uploaded.get(mesh_index)
                    if cached is None:
                        cached = len(self._meshes)
                        mesh = flat_meshes[mesh_index]
                        gl_mesh = upload_mesh(mesh)
                        tex_idx = mesh.base_color_texture_index
                        if tex_idx is not None and tex_idx in image_to_gl_tex:
                            gl_mesh.base_color_texture_id = image_to_gl_tex[tex_idx].texture_id
                        if mesh.mmd_material is not None:
                            self._wire_mmd_textures(gl_mesh, mesh.mmd_material, image_to_gl_tex)
                        self._meshes[cached] = gl_mesh
                        uploaded[mesh_index] = cached
                        self._mesh_index_to_id[mesh_index] = cached
                        self._id_to_mesh_index[cached] = mesh_index
                    self._node_to_mesh.setdefault(id(node), []).append(cached)
                    self._node_mesh_index_to_id[(id(node), mesh_index)] = cached

    def _wire_mmd_textures(
        self,
        gl_mesh: GLMesh,
        material: MMDMaterial,
        image_to_gl_tex: dict[int, GLTexture],
    ) -> None:
        """Resolve sphere / toon texture indices into uploaded GL ids.

        The renderer falls back to the white texture for any index that
        could not be uploaded (decode failure, missing file) — visually
        equivalent to "no sphere / no toon" rather than crashing the draw.
        """
        sphere_idx = material.sphere_texture_index
        if sphere_idx is not None and sphere_idx in image_to_gl_tex:
            gl_mesh.sphere_texture_id = image_to_gl_tex[sphere_idx].texture_id
        toon_idx = material.toon_texture_index
        if toon_idx is not None and toon_idx in image_to_gl_tex:
            toon_tex = image_to_gl_tex[toon_idx]
            gl_mesh.toon_texture_id = toon_tex.texture_id
            # Re-tune the toon ramp to NEAREST + CLAMP_TO_EDGE so cel-shading
            # bands stay crisp instead of being interpolated into a gradient.
            # Idempotent — calling it twice when two materials share a ramp
            # is fine. PMX never reuses a toon image as a base-colour map.
            set_toon_sampler_params(toon_tex.texture_id)

    def _upload_textures(self, imported: ImportedScene) -> dict[int, GLTexture]:
        """Upload every :class:`Texture` in ``imported`` and return ``image_index → GLTexture``.

        Textures bound to a material as the toon ramp or sphere map are
        uploaded with ``srgb=False`` regardless of the importer's flag:
        those textures are lookup tables (Lambert ramp, view-space env
        coord) rather than display colours, so applying the sRGB →
        linear sample conversion would brighten / dim them incorrectly
        under an sRGB-aware framebuffer.
        """
        linear_indices = self._collect_linear_texture_indices(imported)
        out: dict[int, GLTexture] = {}
        for image_index, texture in enumerate(imported.textures):
            use_srgb = bool(texture.srgb) and image_index not in linear_indices
            try:
                gl_tex = upload_texture(texture.pixels, srgb=use_srgb)
            except (GLError, ValueError):
                continue
            self._textures.append(gl_tex)
            out[image_index] = gl_tex
        return out

    def _collect_linear_texture_indices(self, imported: ImportedScene) -> set[int]:
        """Return the set of image indices used as a toon ramp or sphere map.

        Walks every material in ``imported.meshes`` so the upload step
        can pick the right internal format up-front, rather than
        re-uploading after the fact.
        """
        linear: set[int] = set()
        for mesh in imported.meshes:
            material = mesh.mmd_material
            if material is None:
                continue
            toon_idx = material.toon_texture_index
            sphere_idx = material.sphere_texture_index
            if toon_idx is not None:
                linear.add(toon_idx)
            if sphere_idx is not None:
                linear.add(sphere_idx)
        return linear

    def set_secondary_lights(
        self,
        lights: Sequence[
            tuple[tuple[float, float, float], tuple[float, float, float]]
        ],
    ) -> None:
        """Configure up to ``_MAX_SECONDARY_LIGHTS`` HighDef fill lights.

        Each entry is ``((dx, dy, dz), (r, g, b))`` — direction is a
        vector pointing TOWARD the light source (same convention as
        :meth:`set_light`), and colour is RGB scale on the additive
        Lambert contribution. Passing ``[]`` reverts to a single-light
        setup.

        The toon shader treats the primary light (via :meth:`set_light`)
        as the cel-banded contribution with self-shadow; secondaries
        are flat additive linear fill. This matches MMD's HighDef
        convention where a key light defines toon banding and extra
        lights fill specific zones (rim from behind, bounce from
        below).
        """
        trimmed = tuple(lights)[:_MAX_SECONDARY_LIGHTS]
        self._secondary_lights = tuple(
            (tuple(float(c) for c in direction), tuple(float(c) for c in color))
            for direction, color in trimmed
        )

    def apply_highdef_light_preset(self) -> None:
        """Convenience: set the two-light HighDef preset (back rim + front fill)."""
        self.set_secondary_lights([_HIGHDEF_RIM_LIGHT, _HIGHDEF_FILL_LIGHT])

    def set_force_toon_shading(self, enabled: bool) -> None:
        """Route every mesh — including non-MMD imports — through the toon pipeline.

        glTF / OBJ / FBX meshes don't carry an ``MMDMaterial`` and
        normally render through the basic forward path. When this
        toggle is on, the renderer synthesises a generic MMD material
        (almost-white diffuse, thin black edge, 2-band toon ramp) for
        any mesh that doesn't already have one — giving the entire
        scene the MMD cel-shaded look regardless of asset format.

        Off by default so non-MMD scenes keep their native look. Flip
        on per-project when the user wants the MMD aesthetic applied
        to a non-MMD character (the bundled ``herta.glb`` is the
        obvious case — it's a glTF, not a PMX).
        """
        self._force_toon_shading = bool(enabled)

    def set_dqs_enabled(self, enabled: bool) -> None:
        """Toggle dual-quaternion skinning on the toon-skinned path.

        DQS preserves joint volume — twisted shoulders / elbows / wrists
        no longer collapse to a sharp pinch the way LBS does. ON by
        default since pose-driven content (dog crawl, sitting, etc.)
        routinely produces 90°+ joint angles where LBS visibly fails.
        Disable when running pure dance-style motions that stay within
        gentle angles and you want to skip the per-frame bone-matrix-
        to-dual-quaternion conversion in :meth:`_upload_skin_uniforms`.
        """
        self._dqs_enabled = bool(enabled)

    def set_light(
        self, direction: tuple[float, float, float], color: tuple[float, float, float],
    ) -> None:
        """Push a directional light into the toon shader uniforms.

        Direction is normalised at the GPU side (the frag shader does
        ``normalize`` defensively). The integrator typically threads the
        sampled :class:`DirectionalLight` from the VMD scene driver into
        this method once per frame.
        """
        self._light_direction = (
            float(direction[0]), float(direction[1]), float(direction[2]),
        )
        self._light_color = (
            float(color[0]), float(color[1]), float(color[2]),
        )

    def stream_morphed_buffers(
        self,
        positions: np.ndarray | None,
        texcoords: np.ndarray | None,
    ) -> None:
        """Re-upload vertex positions / primary UVs for every uploaded mesh.

        PMX importers alias the same numpy buffer across every material's
        :class:`Mesh`, so the same morphed positions go to all of them.
        Either argument may be ``None`` to skip that channel — useful when
        a frame's active morphs only touched one of the two streams.
        """
        for gl_mesh in self._meshes.values():
            if positions is not None:
                reupload_position_vbo(gl_mesh, positions)
            if texcoords is not None:
                reupload_texcoords_vbo(gl_mesh, texcoords)

    def set_material_overrides(self, overrides: dict[int, MMDMaterial]) -> None:
        """Replace the active material-override map.

        The toon pass falls through to the base :class:`MMDMaterial` for
        meshes whose flat index is absent from the map, so a frame whose
        material morphs all faded back to zero weight automatically gets
        its base materials again as soon as the player calls this with
        an empty dict.
        """
        self._material_overrides = dict(overrides)

    def apply_cloth_state(self, cloth_host: object) -> None:
        """Stream cloth solver positions/normals into the relevant dynamic VBOs.

        Walks ``cloth_host.iter_local_state()`` (duck-typed so this module does
        not depend on the animation layer) and rewrites each bound mesh's
        position + normal VBOs. Must run on the GL-owning thread — call from
        :meth:`Viewport.paintGL` before :meth:`draw`.

        Also seeds the per-frame bone matrix cache from any ``joint_matrix_cache``
        the host exposes — ``cloth_host._update_skin_targets`` has already
        walked every bone parent chain by the time this runs, so the
        renderer can reuse those matrices instead of walking them again
        in the toon / outline / shadow passes that follow.

        When the GPU compute dispatcher is available, passive-skin-deform
        pieces are dispatched here too — the compute shader writes the
        deformed positions + normals straight into the mesh's existing
        VBOs, and :meth:`_stream_cloth_into_meshes` is skipped for those
        pieces (the host already knows not to yield them).
        """
        if not hasattr(cloth_host, "iter_local_state"):
            return
        with frame_section("renderer.apply_cloth"):
            self._adopt_cloth_bone_matrix_cache(cloth_host)
            self._dispatch_gpu_passive_skin(cloth_host)
            self._stream_cloth_into_meshes(cloth_host)

    def _adopt_cloth_bone_matrix_cache(self, cloth_host: object) -> None:
        """Copy the host's per-tick bone matrix cache into this frame's cache."""
        host_cache = getattr(cloth_host, "_last_joint_matrix_cache", None)
        if isinstance(host_cache, dict):
            self._frame_bone_matrix_cache.update(host_cache)

    def _stream_cloth_into_meshes(self, cloth_host: object) -> None:
        """Inner loop of :meth:`apply_cloth_state`, factored for the profile wrapper."""
        for binding, positions_local, normals_local in cloth_host.iter_local_state():
            mesh_id = self._node_mesh_index_to_id.get((id(binding.node), binding.mesh_index))
            if mesh_id is None:
                continue
            gl_mesh = self._meshes.get(mesh_id)
            if gl_mesh is None:
                continue
            reupload_position_vbo(gl_mesh, positions_local)
            reupload_normal_vbo(gl_mesh, normals_local)

    def prepare_gpu_passive_skin(self, cloth_host: object) -> int:
        """Register every passive-skin-deform piece with the compute dispatcher.

        Call once after ``cloth_host.register_imported_scene`` (and after
        every cloth piece has been added to the host) and BEFORE the first
        ``apply_cloth_state``. Walks the host's bindings, finds each
        passive-skin piece with a matching skin follower + GL mesh, and
        allocates its SSBOs in the dispatcher. The host is then told to
        skip its CPU LBS path for those pieces via ``mark_gpu_managed``.

        Returns the count of pieces handed to the GPU. ``0`` means either
        the dispatcher is unavailable (CPU fallback in use) or the scene
        has no passive-skin cloth.
        """
        dispatcher = self._passive_skin_dispatcher
        if dispatcher is None:
            return 0
        registered = 0
        for binding in cloth_host.bindings():
            piece = binding.piece
            if not piece.params.passive_skin_deform:
                continue
            if dispatcher.has_slot(id(piece)):
                continue
            mesh_id = self._node_mesh_index_to_id.get(
                (id(binding.node), binding.mesh_index),
            )
            if mesh_id is None:
                continue
            gl_mesh = self._meshes.get(mesh_id)
            if gl_mesh is None or gl_mesh.normal_vbo_index is None:
                # The compute shader produces normals too — a mesh
                # without a normal VBO can't accept the GPU path
                # cleanly. Stay on CPU for it.
                continue
            follower = cloth_host.follower_for_piece(piece)
            if follower is None or follower.bind_normals is None:
                continue
            dominant = cloth_host._piece_dominant_joint.get(id(piece))  # noqa: SLF001
            if dominant is None:
                continue
            dispatcher.register_piece(
                piece_id=id(piece),
                output_position_vbo=int(gl_mesh.vbos[gl_mesh.position_vbo_index]),
                output_normal_vbo=int(gl_mesh.vbos[gl_mesh.normal_vbo_index]),
                bind_positions=follower.bind_positions,
                bind_normals=follower.bind_normals,
                joints_per_vert=follower.joints_per_vert,
                weights_per_vert=follower.weights_per_vert,
                dominant_joint=dominant,
            )
            cloth_host.mark_gpu_managed(piece)
            registered += 1
        return registered

    def _dispatch_gpu_passive_skin(self, cloth_host: object) -> None:
        """Run the compute shader for every GPU-managed passive-skin piece.

        The cloth host's ``_update_skin_targets`` has already computed the
        joint matrices for every skin this frame; we read them out of
        ``cloth_host.joint_matrices_for_skin`` and feed them to the
        dispatcher. Per-frame state assembled here:

        * world_to_local: the cloth-bound node's inverse world matrix.
        * colliders: the host's current collider list.
        * exclude_bits: packed (MAX_COLLIDERS × BONE_WORDS) bitmask of
          excluded joints per collider, built from the host's per-collider
          filter map.
        """
        dispatcher = self._passive_skin_dispatcher
        if dispatcher is None:
            return
        # Late-binding: scripts can register passive-skin cloth at any
        # point after scene load, and we may have missed them at startup.
        # ``prepare_gpu_passive_skin`` is idempotent on already-registered
        # pieces, so calling it every frame is cheap (one dict lookup per
        # piece) and avoids a "GPU path off until you reload the scene"
        # foot-gun.
        self.prepare_gpu_passive_skin(cloth_host)
        colliders = list(cloth_host.colliders())
        if not colliders:
            return
        filters = [cloth_host.collider_filter_for(c) for c in colliders]
        exclude_bits = build_exclude_bits(filters, len(colliders))
        for binding in cloth_host.bindings():
            piece = binding.piece
            if not piece.enabled or not piece.params.passive_skin_deform:
                continue
            if not cloth_host.is_gpu_managed(piece):
                continue
            follower = cloth_host.follower_for_piece(piece)
            if follower is None:
                continue
            bone_matrices = cloth_host.joint_matrices_for_skin(follower.skin)
            if bone_matrices is None:
                continue
            world_matrix = _world_matrix(binding.node)
            try:
                world_to_local = np.linalg.inv(world_matrix).astype(
                    np.float32, copy=False,
                )
            except np.linalg.LinAlgError:
                continue
            dispatcher.dispatch(
                piece_id=id(piece),
                bone_matrices=bone_matrices,
                world_to_local=world_to_local,
                colliders=colliders,
                exclude_bits=exclude_bits,
            )

    def set_selected_holder(self, holder: Node | None) -> None:
        """Mark ``holder`` (a top-level scene node) as the selected model.

        The next :meth:`draw` runs an extra outline pass over every mesh
        under ``holder`` using ``_SELECTION_EDGE_SIZE`` /
        ``_SELECTION_EDGE_COLOR`` so the user can see which model they
        picked. Passing ``None`` clears the highlight.
        """
        self._selected_holder = holder

    def set_ground_enabled(self, enabled: bool) -> None:
        """Toggle the procedural checker ground + projected shadow pass."""
        self._ground_enabled = bool(enabled)
        # Projected shadow only makes sense when the ground it lands on is
        # being rendered; keep the two in lockstep unless a caller
        # explicitly overrides via :meth:`set_projected_shadow_enabled`.
        self._projected_shadow_enabled = self._ground_enabled

    def set_projected_shadow_enabled(self, enabled: bool) -> None:
        """Toggle the projected ground-shadow pass independently of the ground."""
        self._projected_shadow_enabled = bool(enabled)

    def set_self_shadow_enabled(self, enabled: bool) -> None:
        """Toggle the depth-map self-shadow pass + sampling.

        When off, ``draw`` skips the light-POV depth pass entirely and the
        toon shader receives ``u_shadowEnabled = 0`` — meaning every
        fragment reads as fully lit. Use this for offscreen golden-image
        tests that pin a specific cube render, or for very high-poly
        scenes where the extra depth pass isn't worth it.
        """
        self._self_shadow_enabled = bool(enabled)

    def set_sky_enabled(self, enabled: bool) -> None:
        """Toggle the fullscreen gradient-sky pass.

        When off, the clear colour (dark grey) shows through wherever
        no geometry covers the frame — useful for offscreen tests that
        pin pixel statistics against a flat background.
        """
        self._sky_enabled = bool(enabled)

    def set_srgb_output_enabled(self, enabled: bool) -> None:
        """Toggle sRGB-aware framebuffer output.

        On the editor's default sRGB display this brings colour tone
        closer to a calibrated reference: base-colour textures arrive
        already in sRGB, the GPU decodes them to linear at sample, the
        shader does linear math, and the GPU re-encodes to sRGB on
        write. Smoke tests that pin pixel statistics against a
        pre-sRGB baseline opt out with ``set_srgb_output_enabled(False)``.
        """
        self._srgb_output_enabled = bool(enabled)

    def draw(self, scene: Scene, camera: Camera, viewport_size: tuple[int, int]) -> None:
        """Clear, then walk the scene and draw every node that has a mesh attached."""
        with frame_section("renderer.draw"):
            self._require_program()
            width, height = viewport_size
            if width <= 0 or height <= 0:
                return
            # Reset the per-frame bone matrix cache. The same skin is
            # bone-matrixed up to ~10 times per frame across the toon
            # main pass, outline pass, shadow pass, and projected-shadow
            # pass — caching by ``id(skin)`` collapses those into one
            # walk and saves ~70 ms / frame on a 354-bone rig.
            self._frame_bone_matrix_cache.clear()
            # Per-program "uniform already uploaded this frame" cache —
            # GL keeps uniform values per-program until overwritten, so
            # camera / lights / shadow / bones only need one upload per
            # program even if 30 meshes draw through that program.
            self._program_frame_state.clear()
            # Pre-convert per-frame float32 buffers ONCE. The helpers
            # below pass these directly to ``glUniform*`` — no more
            # per-mesh ``ascontiguousarray`` copies.
            self._light_direction_f32 = np.asarray(
                self._light_direction, dtype=np.float32,
            )
            self._light_color_f32 = np.asarray(
                self._light_color, dtype=np.float32,
            )
            if self._secondary_lights:
                self._secondary_directions_f32 = np.asarray(
                    [d for d, _c in self._secondary_lights], dtype=np.float32,
                )
                self._secondary_colors_f32 = np.asarray(
                    [c for _d, c in self._secondary_lights], dtype=np.float32,
                )
            else:
                self._secondary_directions_f32 = None
                self._secondary_colors_f32 = None
            shadow_pass_active = (
                self._self_shadow_enabled
                and self._shadow_fbo != 0
                and self._shadow_depth_program is not None
            )
            if shadow_pass_active:
                self._light_space_matrix = self._compute_light_space_matrix()
                self._light_space_matrix_f32 = np.ascontiguousarray(
                    self._light_space_matrix, dtype=np.float32,
                )
                with frame_section("renderer.shadow_pass"):
                    self._draw_shadow_pass(scene)
            else:
                self._light_space_matrix_f32 = np.ascontiguousarray(
                    self._light_space_matrix, dtype=np.float32,
                )
            if self._srgb_output_enabled:
                glEnable(GL_FRAMEBUFFER_SRGB)
            else:
                glDisable(GL_FRAMEBUFFER_SRGB)
            glViewport(0, 0, width, height)
            glClearColor(0.08, 0.09, 0.10, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            if self._sky_enabled:
                with frame_section("renderer.sky"):
                    self._draw_sky()

            view = camera.view_matrix()
            proj = camera.projection_matrix(aspect=width / height)
            self._view_f32 = np.ascontiguousarray(view, dtype=np.float32)
            self._proj_f32 = np.ascontiguousarray(proj, dtype=np.float32)
            with frame_section("renderer.scene_nodes"):
                self._draw_scene_nodes(scene, view, proj)
            if self._ground_enabled:
                with frame_section("renderer.ground"):
                    self._draw_ground(view, proj)
            if self._projected_shadow_enabled:
                with frame_section("renderer.projected_shadow"):
                    self._draw_projected_shadows(scene, view, proj)
            with frame_section("renderer.selection_overlay"):
                self._draw_selection_overlay(view, proj)

    def _draw_scene_nodes(self, scene: Scene, view: Mat4, proj: Mat4) -> None:
        for node in scene.root.traverse():
            mesh_ids = self._node_to_mesh.get(id(node))
            if not mesh_ids:
                continue
            skin = self._node_to_skin.get(id(node))
            # When a mesh has BOTH skin AND cloth, the cloth solver already
            # produces world-space positions that account for body motion.
            # Routing those through the skinning shader on top would re-apply
            # the bone transforms (double-positioning) and visibly TEAR the
            # mesh as soon as bones rotate. Prefer the cloth VBO and draw
            # this mesh through the unskinned path. Checked here (not at
            # populate-from-scene time) because ClothComponent is typically
            # attached AFTER scene population, e.g. by ``physics_lite.add_cloth``
            # in the script's ``start()`` hook.
            if skin is not None and any(
                isinstance(c, ClothComponent) for c in node.components
            ):
                skin = None
            for mesh_id in mesh_ids:
                gl_mesh = self._meshes[mesh_id]
                if gl_mesh.mmd_material is not None:
                    self._draw_mmd(node, mesh_id, skin, view, proj)
                elif self._force_toon_shading and self._default_toon_material is not None:
                    self._draw_mmd_synthetic(node, gl_mesh, skin, view, proj)
                elif skin is not None and self._skin_program is not None:
                    self._draw_skinned(mesh_id, skin, view, proj)
                else:
                    self._draw_unskinned(node, mesh_id, view, proj)

    def _draw_mmd_synthetic(
        self, node: Node, gl_mesh: GLMesh, skin: Skin | None,
        view: Mat4, proj: Mat4,
    ) -> None:
        """Draw a non-MMD mesh through the toon pipeline using default settings.

        The synthesised :class:`MMDMaterial` is shared across every
        non-MMD mesh — it doesn't carry per-mesh tints because glTF
        already supplies its own base-colour textures. The default toon
        ramp is bound in place of the mesh's (missing) toon texture so
        the cel banding still appears.
        """
        material = self._default_toon_material
        if material is None:
            return
        if material.has_edge and material.edge_size > 0.0:
            self._draw_outline_with(
                node, gl_mesh, skin,
                edge_size=material.edge_size,
                edge_color=material.edge_color,
                view=view, proj=proj,
            )
        self._draw_toon(
            node, gl_mesh, skin, material, view, proj,
            toon_texture_override=self._default_toon_texture_id,
        )

    def _draw_mmd(
        self, node: Node, mesh_id: int, skin: Skin | None, view: Mat4, proj: Mat4,
    ) -> None:
        """Toon-pipeline draw: optional outline pass, then the toon pass."""
        gl_mesh = self._meshes[mesh_id]
        base_material = gl_mesh.mmd_material
        if base_material is None:
            return
        material = self._effective_material(mesh_id, base_material)
        if material.has_edge and material.edge_size > 0.0:
            self._draw_outline(node, gl_mesh, skin, material, view, proj)
        self._draw_toon(node, gl_mesh, skin, material, view, proj)

    def _effective_material(self, mesh_id: int, base: MMDMaterial) -> MMDMaterial:
        """Return ``base`` shadowed by any active material-morph override."""
        mesh_index = self._id_to_mesh_index.get(mesh_id)
        if mesh_index is None:
            return base
        return self._material_overrides.get(mesh_index, base)

    def _draw_outline(
        self, node: Node, gl_mesh: GLMesh, skin: Skin | None, material: MMDMaterial,
        view: Mat4, proj: Mat4,
    ) -> None:
        """Inverted-hull outline using the material's edge size + colour."""
        self._draw_outline_with(
            node, gl_mesh, skin,
            edge_size=material.edge_size,
            edge_color=material.edge_color,
            view=view, proj=proj,
        )

    def _draw_outline_with(
        self, node: Node, gl_mesh: GLMesh, skin: Skin | None,
        edge_size: float, edge_color: tuple[float, float, float, float],
        view: Mat4, proj: Mat4,
    ) -> None:
        """Inverted-hull outline: front-face culled, expanded along normals.

        Shared between the per-material edge pass and the selection overlay
        — they only differ in ``edge_size`` and ``edge_color``.
        """
        program = self._outline_skin_program if skin is not None else self._outline_program
        if program is None:
            return
        glEnable(GL_CULL_FACE)
        glCullFace(GL_FRONT)
        try:
            with use_program(program.program_id):
                self._set_camera_uniforms(program, view, proj)
                self._set_geometry_uniforms(program, node, skin)
                glUniform1f(program.uniform_location(U_EDGE_SIZE), edge_size)
                glUniform4fv(
                    program.uniform_location(U_EDGE_COLOR),
                    1, np.asarray(edge_color, dtype=np.float32),
                )
                with bind_vao(gl_mesh.vao):
                    glDrawElements(
                        GL_TRIANGLES, gl_mesh.index_count,
                        GL_UNSIGNED_INT, ctypes.c_void_p(0),
                    )
        finally:
            glDisable(GL_CULL_FACE)

    def _draw_sky(self) -> None:
        """Fullscreen gradient sky pass.

        Depth test is disabled (the triangle is on the near plane and
        we don't want it to mask anything anyway) and depth writes are
        off so the rest of the scene's depth values remain whatever
        the clear set them to. The vertex shader synthesises its own
        positions from ``gl_VertexID`` — we bind an empty VAO so a
        Core-Profile driver has something to attach the draw to.
        """
        if self._sky_program is None or self._sky_vao == 0:
            return
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        try:
            with use_program(self._sky_program.program_id):
                glUniform3fv(
                    self._sky_program.uniform_location(U_ZENITH_COLOR), 1,
                    np.asarray(_SKY_ZENITH_COLOR, dtype=np.float32),
                )
                glUniform3fv(
                    self._sky_program.uniform_location(U_HORIZON_COLOR), 1,
                    np.asarray(_SKY_HORIZON_COLOR, dtype=np.float32),
                )
                glUniform3fv(
                    self._sky_program.uniform_location(U_GROUND_COLOR), 1,
                    np.asarray(_SKY_GROUND_COLOR, dtype=np.float32),
                )
                glUniform1f(
                    self._sky_program.uniform_location(U_HORIZON_Y),
                    _SKY_HORIZON_Y,
                )
                glBindVertexArray(self._sky_vao)
                glDrawArrays(GL_TRIANGLES, 0, 3)
                glBindVertexArray(0)
        finally:
            glDepthMask(GL_TRUE)
            glEnable(GL_DEPTH_TEST)

    def _compute_light_space_matrix(self) -> np.ndarray:
        """Build the ortho projection × view matrix for the depth pass.

        The light is placed at ``+_light_direction * _SHADOW_LIGHT_DISTANCE``
        and looks back at the origin, with the orthographic frustum sized
        to ``±_SHADOW_HALF_EXTENT``. Conservative defaults — a single MMD
        character fits comfortably; large stages can override later.
        """
        light_dir = np.asarray(self._light_direction, dtype=np.float32)
        norm = float(np.linalg.norm(light_dir))
        if norm < _DEGENERATE_VEC_EPS:
            light_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            light_dir = light_dir / norm
        eye = light_dir * _SHADOW_LIGHT_DISTANCE
        target = np.zeros(3, dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        # If light is straight up/down, the default ``up`` is parallel
        # and the look-at goes singular — fall back to +Z.
        if abs(float(np.dot(light_dir, up))) > _LOOK_AT_PARALLEL_EPS:
            up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        view = _look_at(eye, target, up)
        proj = _orthographic(
            -_SHADOW_HALF_EXTENT, _SHADOW_HALF_EXTENT,
            -_SHADOW_HALF_EXTENT, _SHADOW_HALF_EXTENT,
            _SHADOW_NEAR, _SHADOW_FAR,
        )
        return (proj @ view).astype(np.float32, copy=False)

    def _draw_shadow_pass(self, scene: Scene) -> None:
        """Render every mesh's depth into ``_shadow_depth_tex`` from light POV.

        Front-face culled so back-face Z is what the toon shader compares
        against — eliminates most shadow acne without per-fragment bias.
        Caller's framebuffer binding is saved before the pass and
        restored on exit, so an offscreen test FBO survives intact.

        Program binding is hoisted out of the per-mesh inner loop: the
        depth pass draws ALL skinned meshes through the same skinned
        program (and the same is true for unskinned), so we only switch
        programs on a transition between the two, instead of doing
        bind / unbind on every mesh.
        """
        previous_fbo = int(glGetIntegerv(GL_FRAMEBUFFER_BINDING))
        glBindFramebuffer(GL_FRAMEBUFFER, self._shadow_fbo)
        glViewport(0, 0, _SHADOW_MAP_SIZE, _SHADOW_MAP_SIZE)
        glClear(GL_DEPTH_BUFFER_BIT)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_FRONT)
        current_program_id = 0
        try:
            for node in scene.root.traverse():
                mesh_ids = self._node_to_mesh.get(id(node))
                if not mesh_ids:
                    continue
                skin = self._node_to_skin.get(id(node))
                program = (
                    self._shadow_depth_skin_program if skin is not None
                    else self._shadow_depth_program
                )
                if program is None:
                    continue
                if program.program_id != current_program_id:
                    glUseProgram(program.program_id)
                    current_program_id = program.program_id
                # Light-space matrix is frame-constant — cache per program
                # through the same state machine the toon pass uses.
                state = self._program_state(program)
                if not state.shadow_uploaded:
                    state.shadow_uploaded = True
                    light_space_f32 = (
                        self._light_space_matrix_f32
                        if self._light_space_matrix_f32 is not None
                        else np.ascontiguousarray(
                            self._light_space_matrix, dtype=np.float32,
                        )
                    )
                    glUniformMatrix4fv(
                        program.uniform_location(U_LIGHT_SPACE_MATRIX),
                        1, GL_TRUE, light_space_f32,
                    )
                self._set_geometry_uniforms(program, node, skin)
                for mesh_id in mesh_ids:
                    gl_mesh = self._meshes[mesh_id]
                    glBindVertexArray(gl_mesh.vao)
                    glDrawElements(
                        GL_TRIANGLES, gl_mesh.index_count,
                        GL_UNSIGNED_INT, ctypes.c_void_p(0),
                    )
        finally:
            glBindVertexArray(0)
            if current_program_id != 0:
                glUseProgram(0)
            glCullFace(GL_BACK)
            glDisable(GL_CULL_FACE)
            glBindFramebuffer(GL_FRAMEBUFFER, previous_fbo)

    def _select_toon_program(self, skin: Skin | None) -> Program | None:
        """Pick the right toon program for the current skin + DQS toggle.

        Unskinned meshes always go through the non-skinned toon program.
        Skinned meshes pick the DQS variant when the toggle is on AND
        the DQS program compiled successfully — otherwise they fall
        back to the LBS skinned program, so a missing DQS shader
        gracefully degrades rather than dropping the mesh.
        """
        if skin is None:
            return self._toon_program
        if self._dqs_enabled and self._toon_skin_dqs_program is not None:
            return self._toon_skin_dqs_program
        return self._toon_skin_program

    def _bind_secondary_lights(self, program: Program) -> None:
        """Upload the HighDef fill / rim lights to the toon program.

        Always writes ``u_secondaryLightCount`` so the shader's loop
        condition is well-defined even when zero secondaries are
        configured. The arrays themselves are only uploaded up to the
        active count to keep the per-draw cost flat at low light
        counts. Cached once per program per frame — subsequent meshes
        drawing through the same program skip the whole helper.
        """
        state = self._program_state(program)
        if state.lights_uploaded:
            return
        count_loc = program.uniform_location(U_SECONDARY_LIGHT_COUNT)
        if count_loc < 0:
            return
        state.lights_uploaded = True
        count = len(self._secondary_lights)
        glUniform1i(count_loc, count)
        if count == 0:
            return
        directions = (
            self._secondary_directions_f32
            if self._secondary_directions_f32 is not None
            else np.asarray(
                [d for d, _color in self._secondary_lights], dtype=np.float32,
            )
        )
        colors = (
            self._secondary_colors_f32
            if self._secondary_colors_f32 is not None
            else np.asarray(
                [c for _direction, c in self._secondary_lights], dtype=np.float32,
            )
        )
        glUniform3fv(
            program.uniform_location(U_SECONDARY_LIGHT_DIRECTIONS),
            count, directions,
        )
        glUniform3fv(
            program.uniform_location(U_SECONDARY_LIGHT_COLORS),
            count, colors,
        )

    def _bind_shadow_uniforms(self, program: Program) -> None:
        """Push self-shadow uniforms into the toon program.

        Toon and toon_skinned share the same uniform names; this helper
        is called from :meth:`_draw_toon` once per draw so the depth
        compare in the frag shader picks up the right matrix + sampler.
        """
        state = self._program_state(program)
        if state.shadow_uploaded:
            return
        light_space_loc = program.uniform_location(U_LIGHT_SPACE_MATRIX)
        if light_space_loc < 0:
            return
        state.shadow_uploaded = True
        active = 1 if (
            self._self_shadow_enabled and self._shadow_depth_tex != 0
        ) else 0
        light_space_f32 = (
            self._light_space_matrix_f32
            if self._light_space_matrix_f32 is not None
            else np.ascontiguousarray(self._light_space_matrix, dtype=np.float32)
        )
        glUniformMatrix4fv(light_space_loc, 1, GL_TRUE, light_space_f32)
        glUniform1i(program.uniform_location(U_SHADOW_ENABLED), active)
        glUniform1f(program.uniform_location(U_SHADOW_STRENGTH), _SHADOW_STRENGTH)
        glUniform1i(program.uniform_location(U_SHADOW_MAP), _TEX_UNIT_SHADOW)
        if active == 1:
            glActiveTexture(GL_TEXTURE0 + _TEX_UNIT_SHADOW)
            glBindTexture(GL_TEXTURE_2D, self._shadow_depth_tex)
            glActiveTexture(GL_TEXTURE0)

    def _draw_ground(self, view: Mat4, proj: Mat4) -> None:
        """Draw the procedural MMD-style checkered floor at ``y=0``.

        Single quad covering ``±_GROUND_HALF_EXTENT``; the fragment shader
        does the checker pattern from world XZ and fades the alpha near
        the horizon so the floor blends into the clear colour instead of
        ending in a hard line. Alpha blending is on so the fade survives.
        """
        program = self._ground_program
        if program is None or self._ground_vao == 0:
            return
        identity = np.eye(4, dtype=np.float32)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        try:
            with use_program(program.program_id):
                self._set_camera_uniforms(program, view, proj)
                glUniformMatrix4fv(
                    program.uniform_location(U_MODEL_MATRIX),
                    1, GL_TRUE, identity,
                )
                glUniform1f(program.uniform_location(U_CELL_SIZE), _GROUND_CELL_SIZE)
                glUniform4fv(
                    program.uniform_location(U_COLOR_A), 1,
                    np.asarray(_GROUND_COLOR_A, dtype=np.float32),
                )
                glUniform4fv(
                    program.uniform_location(U_COLOR_B), 1,
                    np.asarray(_GROUND_COLOR_B, dtype=np.float32),
                )
                glUniform1f(program.uniform_location(U_FADE_START), _GROUND_FADE_START)
                glUniform1f(program.uniform_location(U_FADE_END), _GROUND_FADE_END)
                glBindVertexArray(self._ground_vao)
                glDrawElements(
                    GL_TRIANGLES, self._ground_index_count,
                    GL_UNSIGNED_INT, ctypes.c_void_p(0),
                )
                glBindVertexArray(0)
        finally:
            glDisable(GL_BLEND)

    def _draw_projected_shadows(
        self, scene: Scene, view: Mat4, proj: Mat4,
    ) -> None:
        """Flatten every mesh's silhouette onto the ground plane.

        Cheap stand-in for a depth-map shadow: the vertex shader projects
        each vertex along ``-u_lightDirection`` onto ``y=_GROUND_Y_LEVEL``
        plus a 1 mm offset so the shadow wins the z-fight with the ground
        quad. Depth write is off so a single vert that overshoots the
        ground (anchor below the floor, foot a hair under-rig, …) can't
        seed an artefact in later passes; depth TEST is still on so the
        shadow is hidden behind 3D obstacles in front of the ground.
        """
        if self._shadow_proj_program is None or self._shadow_proj_skin_program is None:
            return
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)
        current_program_id = 0
        shadow_color_f32 = np.asarray(_PROJECTED_SHADOW_COLOR, dtype=np.float32)
        try:
            for node in scene.root.traverse():
                mesh_ids = self._node_to_mesh.get(id(node))
                if not mesh_ids:
                    continue
                skin = self._node_to_skin.get(id(node))
                program = (
                    self._shadow_proj_skin_program if skin is not None
                    else self._shadow_proj_program
                )
                if program.program_id != current_program_id:
                    glUseProgram(program.program_id)
                    current_program_id = program.program_id
                # Per-program one-time constants. ``camera_uploaded`` and
                # ``lights_uploaded`` are the same flags the toon pass
                # uses — programs are distinct so the flags don't collide.
                self._set_camera_uniforms(program, view, proj)
                state = self._program_state(program)
                if not state.lights_uploaded:
                    state.lights_uploaded = True
                    glUniform3fv(
                        program.uniform_location(U_LIGHT_DIRECTION), 1,
                        self._light_direction_f32 if self._light_direction_f32 is not None
                        else np.asarray(self._light_direction, dtype=np.float32),
                    )
                    glUniform1f(program.uniform_location(U_GROUND_Y), _GROUND_Y_LEVEL)
                    glUniform4fv(
                        program.uniform_location(U_SHADOW_COLOR), 1, shadow_color_f32,
                    )
                self._set_geometry_uniforms(program, node, skin)
                for mesh_id in mesh_ids:
                    gl_mesh = self._meshes[mesh_id]
                    glBindVertexArray(gl_mesh.vao)
                    glDrawElements(
                        GL_TRIANGLES, gl_mesh.index_count,
                        GL_UNSIGNED_INT, ctypes.c_void_p(0),
                    )
        finally:
            glBindVertexArray(0)
            if current_program_id != 0:
                glUseProgram(0)
            glDepthMask(GL_TRUE)
            glDisable(GL_BLEND)

    def _draw_selection_overlay(self, view: Mat4, proj: Mat4) -> None:
        """Re-outline every mesh under :attr:`_selected_holder` for selection feedback.

        Runs after the main scene pass so the highlight sits on top of any
        per-material outline. Uses the inverted-hull outline shader at a
        fixed thicker ``_SELECTION_EDGE_SIZE`` so even materials with no
        edge enabled get a visible silhouette while selected.
        """
        holder = self._selected_holder
        if holder is None:
            return
        for node in holder.traverse():
            mesh_ids = self._node_to_mesh.get(id(node))
            if not mesh_ids:
                continue
            skin = self._node_to_skin.get(id(node))
            for mesh_id in mesh_ids:
                gl_mesh = self._meshes[mesh_id]
                self._draw_outline_with(
                    node, gl_mesh, skin,
                    edge_size=_SELECTION_EDGE_SIZE,
                    edge_color=_SELECTION_EDGE_COLOR,
                    view=view, proj=proj,
                )

    def _draw_toon(
        self, node: Node, gl_mesh: GLMesh, skin: Skin | None, material: MMDMaterial,
        view: Mat4, proj: Mat4,
        toon_texture_override: int | None = None,
    ) -> None:
        """Toon-shaded forward pass — Lambert × ramp + sphere composite + Blinn-Phong.

        ``toon_texture_override`` lets the synthetic-MMD path pass in the
        engine's default 2-band toon ramp without having to mutate the
        cached :class:`GLMesh`. Pass ``None`` (default) to use the mesh's
        own ramp.
        """
        program = self._select_toon_program(skin)
        if program is None:
            return
        if not material.is_double_sided:
            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)
        try:
            with use_program(program.program_id):
                self._set_camera_uniforms(program, view, proj)
                self._set_geometry_uniforms(program, node, skin)
                self._bind_toon_textures(
                    program, gl_mesh, material,
                    toon_texture_override=toon_texture_override,
                )
                self._bind_toon_scalars(program, material)
                self._bind_secondary_lights(program)
                self._bind_shadow_uniforms(program)
                with bind_vao(gl_mesh.vao):
                    glDrawElements(
                        GL_TRIANGLES, gl_mesh.index_count,
                        GL_UNSIGNED_INT, ctypes.c_void_p(0),
                    )
        finally:
            glDisable(GL_CULL_FACE)

    def _bone_matrices_for(self, skin: Skin) -> np.ndarray:
        """Return ``skin``'s bone matrices, cached for the rest of the frame.

        Backs every per-draw bone-matrix request via the renderer's
        per-frame cache so the same skin only computes its parent
        chains once even though the toon, outline, shadow and projected-
        shadow passes all sample it.
        """
        cached = self._frame_bone_matrix_cache.get(id(skin))
        if cached is not None:
            return cached
        computed = _compute_bone_matrices(skin)
        self._frame_bone_matrix_cache[id(skin)] = computed
        return computed

    def _set_geometry_uniforms(
        self, program: Program, node: Node, skin: Skin | None,
    ) -> None:
        """Push per-draw geometry transforms — model+normal for non-skinned,
        the bone matrix array for skinned. Both pipelines share view/proj."""
        if skin is None:
            model_matrix = _world_matrix(node)
            glUniformMatrix4fv(
                program.uniform_location(U_MODEL_MATRIX), 1, GL_TRUE,
                np.ascontiguousarray(model_matrix, dtype=np.float32),
            )
            normal_matrix = _normal_matrix_from(model_matrix)
            glUniformMatrix3fv(
                program.uniform_location(U_NORMAL_MATRIX), 1, GL_TRUE,
                np.ascontiguousarray(normal_matrix, dtype=np.float32),
            )
            return
        # Bone matrices are per-skin, not per-mesh: many meshes can be
        # weighted to the same skeleton (clothing + body + hair all share
        # one armature). Once we've uploaded the matrix palette for skin
        # X to program P this frame, every subsequent mesh that draws
        # through (P, X) reads the already-bound value — skip the
        # ~6 KB upload. Saves ~1.5 ms / frame on the Herta rig where
        # 5+ meshes share one 354-bone skin across 3 passes.
        state = self._program_state(program)
        skin_key = id(skin)
        if skin_key in state.skin_uploaded:
            return
        state.skin_uploaded.add(skin_key)
        bone_matrices = self._bone_matrices_for(skin)
        bone_matrices_loc = program.uniform_location(U_BONE_MATRICES)
        if bone_matrices_loc >= 0:
            glUniformMatrix4fv(
                bone_matrices_loc,
                bone_matrices.shape[0], GL_TRUE,
                np.ascontiguousarray(bone_matrices, dtype=np.float32),
            )
            return
        # DQS program: convert bone matrices to dual quaternions and
        # upload the real + dual halves as separate vec4 arrays.
        dual_quats = matrices_to_dual_quaternions(bone_matrices)
        count = dual_quats.shape[0]
        glUniform4fv(
            program.uniform_location(U_BONE_DQ_REAL),
            count, np.ascontiguousarray(dual_quats[:, :4], dtype=np.float32),
        )
        glUniform4fv(
            program.uniform_location(U_BONE_DQ_DUAL),
            count, np.ascontiguousarray(dual_quats[:, 4:], dtype=np.float32),
        )

    def _bind_toon_textures(
        self, program: Program, gl_mesh: GLMesh, material: MMDMaterial,
        toon_texture_override: int | None = None,
    ) -> None:
        """Bind the three toon-pass textures (albedo / sphere / toon ramp).

        Each unit always has a real texture bound; uniform fall-throughs go
        to the white fallback so unbound samplers do not produce undefined
        sampler values that would surface as driver-specific noise.

        ``toon_texture_override`` (when given) supersedes the mesh's own
        toon ramp — used by the synthetic-MMD path so non-MMD imports
        get the engine's default 2-band cel ramp.
        """
        fallback_id = self._white_fallback.texture_id if self._white_fallback else 0
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, gl_mesh.base_color_texture_id or fallback_id)
        glUniform1i(program.uniform_location(U_BASE_COLOR_TEX), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, gl_mesh.sphere_texture_id or fallback_id)
        glUniform1i(program.uniform_location(U_SPHERE_TEX), 1)

        glActiveTexture(GL_TEXTURE2)
        toon_id = (
            toon_texture_override
            if toon_texture_override is not None
            else (gl_mesh.toon_texture_id or fallback_id)
        )
        glBindTexture(GL_TEXTURE_2D, toon_id)
        glUniform1i(program.uniform_location(U_TOON_TEX), 2)

        glUniform1i(
            program.uniform_location(U_SPHERE_MODE),
            int(material.sphere_mode),
        )

    def _bind_toon_scalars(self, program: Program, material: MMDMaterial) -> None:
        # Material colours change per mesh (different materials per mesh)
        # so we always upload those. Light direction / colour are scene
        # constants per frame, but the toon program writes them into the
        # same shared uniform — uploading the float32 cache here is a
        # single glUniform3fv instead of a per-material asarray.
        glUniform4fv(
            program.uniform_location(U_BASE_COLOR), 1,
            np.asarray(material.diffuse, dtype=np.float32),
        )
        glUniform3fv(
            program.uniform_location(U_SPECULAR), 1,
            np.asarray(material.specular, dtype=np.float32),
        )
        glUniform1f(
            program.uniform_location(U_SPECULAR_POWER),
            float(material.specular_power),
        )
        glUniform3fv(
            program.uniform_location(U_AMBIENT), 1,
            np.asarray(material.ambient, dtype=np.float32),
        )
        glUniform3fv(
            program.uniform_location(U_LIGHT_DIRECTION), 1,
            self._light_direction_f32 if self._light_direction_f32 is not None
            else np.asarray(self._light_direction, dtype=np.float32),
        )
        glUniform3fv(
            program.uniform_location(U_LIGHT_COLOR), 1,
            self._light_color_f32 if self._light_color_f32 is not None
            else np.asarray(self._light_color, dtype=np.float32),
        )

    def _draw_unskinned(self, node: Node, mesh_id: int, view: Mat4, proj: Mat4) -> None:
        program = self._require_program()
        gl_mesh = self._meshes[mesh_id]
        model_matrix = _world_matrix(node)
        with use_program(program.program_id):
            self._set_camera_uniforms(program, view, proj)
            loc_model = program.uniform_location(U_MODEL_MATRIX)
            loc_normal = program.uniform_location(U_NORMAL_MATRIX)
            glUniformMatrix4fv(
                loc_model, 1, GL_TRUE,
                np.ascontiguousarray(model_matrix, dtype=np.float32),
            )
            normal_matrix = _normal_matrix_from(model_matrix)
            glUniformMatrix3fv(
                loc_normal, 1, GL_TRUE,
                np.ascontiguousarray(normal_matrix, dtype=np.float32),
            )
            self._bind_material_uniforms(program, gl_mesh)
            with bind_vao(gl_mesh.vao):
                glDrawElements(
                    GL_TRIANGLES, gl_mesh.index_count, GL_UNSIGNED_INT, ctypes.c_void_p(0),
                )

    def _draw_skinned(self, mesh_id: int, skin: Skin, view: Mat4, proj: Mat4) -> None:
        program = self._skin_program
        if program is None:
            return
        gl_mesh = self._meshes[mesh_id]
        bone_matrices = self._bone_matrices_for(skin)
        with use_program(program.program_id):
            self._set_camera_uniforms(program, view, proj)
            loc_bones = program.uniform_location(U_BONE_MATRICES)
            glUniformMatrix4fv(
                loc_bones, bone_matrices.shape[0], GL_TRUE,
                np.ascontiguousarray(bone_matrices, dtype=np.float32),
            )
            self._bind_material_uniforms(program, gl_mesh)
            with bind_vao(gl_mesh.vao):
                glDrawElements(
                    GL_TRIANGLES, gl_mesh.index_count, GL_UNSIGNED_INT, ctypes.c_void_p(0),
                )

    def _set_camera_uniforms(self, program: Program, view: Mat4, proj: Mat4) -> None:
        state = self._program_state(program)
        if state.camera_uploaded:
            return
        state.camera_uploaded = True
        # ``draw()`` populated ``_view_f32`` / ``_proj_f32`` once at the
        # top of the frame; fall through to ascontiguousarray only if a
        # caller invoked this helper outside the normal draw loop
        # (e.g. an offscreen test) where the cache is None.
        view_f32 = self._view_f32 if self._view_f32 is not None else (
            np.ascontiguousarray(view, dtype=np.float32)
        )
        proj_f32 = self._proj_f32 if self._proj_f32 is not None else (
            np.ascontiguousarray(proj, dtype=np.float32)
        )
        glUniformMatrix4fv(
            program.uniform_location(U_VIEW_MATRIX), 1, GL_TRUE, view_f32,
        )
        glUniformMatrix4fv(
            program.uniform_location(U_PROJECTION_MATRIX), 1, GL_TRUE, proj_f32,
        )

    def _program_state(self, program: Program) -> _ProgramFrameState:
        """Return (lazily creating) the per-frame state for ``program``."""
        state = self._program_frame_state.get(program.program_id)
        if state is None:
            state = _ProgramFrameState()
            self._program_frame_state[program.program_id] = state
        return state

    def apply_effect_chain(
        self,
        executor: EffectChainExecutor,
        chain: EffectChain,
        viewport_size: tuple[int, int],
        main_color_texture: int,
        *,
        default_framebuffer: int = 0,
    ) -> None:
        """Run ``chain`` through ``executor`` against an FBO ping-pong pool.

        First call (or any call where ``viewport_size`` changes) (re-)allocates
        the two FBOs. ``main_color_texture`` is the GL handle of the renderer's
        main-pass output; subsequent passes can read it as ``main_color`` and
        chain via the ``"result"`` source. After every enabled pass has fired,
        the latest texture is blitted into ``default_framebuffer`` (window
        framebuffer = 0, or a Qt-supplied FBO id under ``QOpenGLWidget``).
        """
        width, height = viewport_size
        if width <= 0 or height <= 0:
            return
        if self._effect_ping_pong is None:
            self._effect_ping_pong = EffectPingPong()
        ping_pong = self._effect_ping_pong
        if not ping_pong._allocated or ping_pong.width != width or ping_pong.height != height:  # noqa: SLF001
            ping_pong.allocate(width, height)
        ping_pong.begin_chain(main_color_texture=int(main_color_texture))
        executor.run(
            chain,
            bind_input=ping_pong.bind_input,
            bind_output=ping_pong.bind_output,
            draw_quad=ping_pong.draw_quad,
            before_pass=ping_pong.before_pass,
        )
        ping_pong.present(default_framebuffer=default_framebuffer)

    def release_effect_resources(self) -> None:
        """Free the ping-pong FBO pool. Call from the GL thread on shutdown."""
        if self._effect_ping_pong is not None:
            self._effect_ping_pong.deallocate()
            self._effect_ping_pong = None

    def world_aabb_of_subtree(
        self, root: Node,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Conservative world-space AABB covering every uploaded mesh under ``root``.

        Walks ``root`` (and descendants) for nodes with mesh bindings; for each
        bound :class:`GLMesh`, transforms its 8 mesh-local AABB corners by the
        node's world matrix and folds the result into a running min/max. Used
        by interactive picking — see :meth:`Viewport._pick_holder`.
        """
        big = float("inf")
        mn = np.array([big, big, big], dtype=np.float32)
        mx = np.array([-big, -big, -big], dtype=np.float32)
        found = False
        for node in root.traverse():
            mesh_ids = self._node_to_mesh.get(id(node))
            if not mesh_ids:
                continue
            world_m = _world_matrix(node)
            for mesh_id in mesh_ids:
                gl_mesh = self._meshes[mesh_id]
                corners = _aabb_corners(gl_mesh.aabb_min, gl_mesh.aabb_max)
                ones = np.ones((corners.shape[0], 1), dtype=np.float32)
                world_corners = (world_m @ np.hstack([corners, ones]).T).T[:, :3]
                mn = np.minimum(mn, world_corners.min(axis=0))
                mx = np.maximum(mx, world_corners.max(axis=0))
                found = True
        return (mn, mx) if found else None

    def _bind_material_uniforms(self, program: Program, gl_mesh: GLMesh) -> None:
        loc_color = program.uniform_location(U_BASE_COLOR)
        loc_tex = program.uniform_location(U_BASE_COLOR_TEX)
        color = gl_mesh.base_color if gl_mesh.base_color is not None else self.base_color
        glUniform4fv(loc_color, 1, np.asarray(color, dtype=np.float32))
        glActiveTexture(GL_TEXTURE0)
        tex_id = gl_mesh.base_color_texture_id
        if tex_id is None and self._white_fallback is not None:
            tex_id = self._white_fallback.texture_id
        if tex_id is not None:
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glUniform1i(loc_tex, 0)

    def _require_program(self) -> Program:
        if self._program is None:
            raise GLError("Renderer.initialize() must be called before draw()")
        return self._program


def _world_matrix(node: Node) -> Mat4:
    matrix = node.transform.to_matrix()
    parent = node.parent
    while parent is not None:
        matrix = parent.transform.to_matrix() @ matrix
        parent = parent.parent
    return matrix.astype(np.float32, copy=False)


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Right-handed look-at matrix (camera-space) matching the GL convention.

    Used to point the shadow-pass camera from ``eye`` toward ``target``;
    factored out so the camera module's own ``view_matrix`` stays
    untouched.
    """
    forward = target - eye
    norm = float(np.linalg.norm(forward))
    if norm < _DEGENERATE_VEC_EPS:
        return np.eye(4, dtype=np.float32)
    f = forward / norm
    s = np.cross(f, up)
    s_norm = float(np.linalg.norm(s))
    if s_norm < _DEGENERATE_VEC_EPS:
        return np.eye(4, dtype=np.float32)
    s = s / s_norm
    u = np.cross(s, f)
    out = np.eye(4, dtype=np.float32)
    out[0, :3] = s
    out[1, :3] = u
    out[2, :3] = -f
    out[0, 3] = -float(np.dot(s, eye))
    out[1, 3] = -float(np.dot(u, eye))
    out[2, 3] = float(np.dot(f, eye))
    return out


def _orthographic(
    left: float, right: float, bottom: float, top: float,
    near: float, far: float,
) -> np.ndarray:
    """Right-handed orthographic projection matrix matching GL clip space.

    The shadow map's depth values come from this projection's Z, so the
    near / far planes need to bracket the scene tightly enough for the
    24-bit depth texture to give clean comparisons.
    """
    out = np.eye(4, dtype=np.float32)
    out[0, 0] = 2.0 / (right - left)
    out[1, 1] = 2.0 / (top - bottom)
    out[2, 2] = -2.0 / (far - near)
    out[0, 3] = -(right + left) / (right - left)
    out[1, 3] = -(top + bottom) / (top - bottom)
    out[2, 3] = -(far + near) / (far - near)
    return out


def _normal_matrix_from(model: Mat4) -> np.ndarray:
    """Inverse-transpose of the upper 3x3 of ``model``; identity for non-invertible blocks."""
    upper = np.asarray(model[:3, :3], dtype=np.float32)
    try:
        return np.linalg.inv(upper).T.astype(np.float32, copy=False)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float32)


def _find_skin(node: Node) -> Skin | None:
    """Return the :class:`Skin` referenced by a :class:`SkinRefComponent` on ``node``."""
    for component in node.components:
        if isinstance(component, SkinRefComponent) and component.skin is not None:
            return component.skin  # type: ignore[return-value]
    return None


def _aabb_corners(
    aabb_min: tuple[float, float, float],
    aabb_max: tuple[float, float, float],
) -> np.ndarray:
    """Return the eight corners of an axis-aligned bounding box as ``(8, 3)``."""
    x0, y0, z0 = aabb_min
    x1, y1, z1 = aabb_max
    return np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x0, y1, z0], [x1, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x0, y1, z1], [x1, y1, z1],
        ],
        dtype=np.float32,
    )


def _compute_bone_matrices(skin: Skin) -> np.ndarray:
    """Return ``(J, 4, 4)`` matrices = joint world × inverse-bind, capped at MAX_BONES.

    Joint world matrices are computed by walking each joint's parent chain to
    the scene root, so any holder transform applied above the imported scene
    automatically folds into the bone matrices.

    A per-call ``world_cache`` keyed on ``id(node)`` memoises ancestor world
    matrices, turning the otherwise quadratic ``joints × depth`` chain walk
    into O(joints + unique ancestors). For a 135-joint VRoid rig the naive
    version did ~1350 4×4 matmuls per frame; with caching it drops to ~140,
    which is the difference between 60 fps and stuttering on the CPU side.
    """
    joints = skin.joints
    inverse_binds = skin.inverse_bind_matrices
    declared = min(len(joints), inverse_binds.shape[0])
    if declared > _MAX_BONES:
        skin_id = id(skin)
        if skin_id not in _warned_oversize_skins:
            _warned_oversize_skins.add(skin_id)
            _logger.warning(
                "skin %r has %d joints but renderer caps at %d; joints beyond the "
                "cap are clamped to identity and any vertex weighted to them will "
                "appear deformed. Bump _MAX_BONES + MAX_BONES in the skinned "
                "shaders to fit this rig.",
                skin.name, declared, _MAX_BONES,
            )
    joint_count = min(declared, _MAX_BONES)
    world_cache: dict[int, np.ndarray] = {}

    def world(node: Node) -> np.ndarray:
        cached = world_cache.get(id(node))
        if cached is not None:
            return cached
        local = node.transform.to_matrix()
        parent = node.parent
        result = local if parent is None else world(parent) @ local
        world_cache[id(node)] = result
        return result

    # Stack all joint world matrices into a single (J, 4, 4) buffer, then
    # let numpy issue ONE batched matmul against the inverse-bind stack.
    # Replaces the previous per-joint Python-level ``@`` (which calls
    # numpy's overhead-heavy single-mat4 matmul J times — 354 times for
    # the Herta rig — and dominated frame time at ~0.8 ms / call).
    bone_worlds = np.empty((joint_count, 4, 4), dtype=np.float32)
    eye4 = np.eye(4, dtype=np.float32)
    for i in range(joint_count):
        joint = joints[i]
        bone_worlds[i] = (
            world(joint) if isinstance(joint, Node) else eye4
        )
    return np.matmul(
        bone_worlds, np.asarray(inverse_binds[:joint_count], dtype=np.float32),
    )
