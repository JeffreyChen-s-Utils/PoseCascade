"""Forward renderer: walks a :class:`~posecascade.scene.scene.Scene` and draws it.

The renderer owns the basic forward shader program and the GPU mesh cache.
``draw`` is meant to be called from :meth:`~posecascade.ui.viewport.Viewport.paintGL`
on the GL-owning thread; cross-thread calls trip
:meth:`~posecascade.gl.context.GLContext.assert_owned`.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from OpenGL.GL import (
    GL_BACK,
    GL_COLOR_BUFFER_BIT,
    GL_CULL_FACE,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_FRONT,
    GL_TEXTURE0,
    GL_TEXTURE1,
    GL_TEXTURE2,
    GL_TEXTURE_2D,
    GL_TRIANGLES,
    GL_TRUE,
    GL_UNSIGNED_INT,
    glActiveTexture,
    glBindTexture,
    glClear,
    glClearColor,
    glCullFace,
    glDisable,
    glDrawElements,
    glEnable,
    glUniform1f,
    glUniform1i,
    glUniform3fv,
    glUniform4fv,
    glUniformMatrix3fv,
    glUniformMatrix4fv,
    glViewport,
)

from posecascade.assets.types import ImportedScene, Mesh, Skin
from posecascade.errors import GLError
from posecascade.gl.binding import bind_vao, use_program
from posecascade.gl.mesh_uploader import (
    GLMesh,
    reupload_normal_vbo,
    reupload_position_vbo,
    reupload_texcoords_vbo,
    upload_mesh,
)
from posecascade.gl.shader import Program, compile_program
from posecascade.gl.texture import GLTexture, make_white_fallback, upload_texture
from posecascade.gl.uniforms import (
    U_AMBIENT,
    U_BASE_COLOR,
    U_BASE_COLOR_TEX,
    U_BONE_MATRICES,
    U_EDGE_COLOR,
    U_EDGE_SIZE,
    U_LIGHT_COLOR,
    U_LIGHT_DIRECTION,
    U_MODEL_MATRIX,
    U_NORMAL_MATRIX,
    U_PROJECTION_MATRIX,
    U_SPECULAR,
    U_SPECULAR_POWER,
    U_SPHERE_MODE,
    U_SPHERE_TEX,
    U_TOON_TEX,
    U_VIEW_MATRIX,
)
from posecascade.render.camera import Camera
from posecascade.render.effects.chain import EffectChain
from posecascade.render.effects.executor import EffectChainExecutor
from posecascade.render.effects.ping_pong import EffectPingPong
from posecascade.render.material import MMDMaterial
from posecascade.scene.component import MeshRefComponent, SkinRefComponent
from posecascade.scene.node import Node
from posecascade.scene.scene import Scene
from posecascade.utils.logging import get_logger
from posecascade.utils.math3d import Mat4

_logger = get_logger(__name__)
_warned_oversize_skins: set[int] = set()

_DEFAULT_BASE_COLOR = (0.8, 0.8, 0.85, 1.0)
_MAX_BONES = 256  # must match skinned.vert's MAX_BONES define
_DEFAULT_LIGHT_DIRECTION = (0.3, 0.7, 0.6)
_DEFAULT_LIGHT_COLOR = (1.0, 1.0, 1.0)


@dataclass
class Renderer:
    """Owns the shader programs and the mesh-id → :class:`GLMesh` cache."""

    shaders_root: Path
    base_color: tuple[float, float, float, float] = _DEFAULT_BASE_COLOR
    _program: Program | None = field(default=None, init=False)
    _skin_program: Program | None = field(default=None, init=False)
    _toon_program: Program | None = field(default=None, init=False)
    _toon_skin_program: Program | None = field(default=None, init=False)
    _outline_program: Program | None = field(default=None, init=False)
    _outline_skin_program: Program | None = field(default=None, init=False)
    _meshes: dict[int, GLMesh] = field(default_factory=dict, init=False)
    _node_to_mesh: dict[int, list[int]] = field(default_factory=dict, init=False)
    _node_to_skin: dict[int, Skin] = field(default_factory=dict, init=False)
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
    # Active material overrides keyed on the same flat-mesh index. ``apply``
    # of a MorphSnapshot replaces this entire dict, so a morph that fades
    # back to zero weight automatically stops shadowing its base material.
    _material_overrides: dict[int, MMDMaterial] = field(default_factory=dict, init=False)
    # Lazily allocated post-effect ping-pong; only instantiated once the
    # caller requests :meth:`apply_effect_chain`.
    _effect_ping_pong: EffectPingPong | None = field(default=None, init=False)

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
        glEnable(GL_DEPTH_TEST)
        self._white_fallback = make_white_fallback()

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
            gl_mesh.toon_texture_id = image_to_gl_tex[toon_idx].texture_id

    def _upload_textures(self, imported: ImportedScene) -> dict[int, GLTexture]:
        """Upload every :class:`Texture` in ``imported`` and return ``image_index → GLTexture``."""
        out: dict[int, GLTexture] = {}
        for image_index, texture in enumerate(imported.textures):
            try:
                gl_tex = upload_texture(texture.pixels, srgb=texture.srgb)
            except (GLError, ValueError):
                continue
            self._textures.append(gl_tex)
            out[image_index] = gl_tex
        return out

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
        """
        if not hasattr(cloth_host, "iter_local_state"):
            return
        for binding, positions_local, normals_local in cloth_host.iter_local_state():
            mesh_id = self._node_mesh_index_to_id.get((id(binding.node), binding.mesh_index))
            if mesh_id is None:
                continue
            gl_mesh = self._meshes.get(mesh_id)
            if gl_mesh is None:
                continue
            reupload_position_vbo(gl_mesh, positions_local)
            reupload_normal_vbo(gl_mesh, normals_local)

    def draw(self, scene: Scene, camera: Camera, viewport_size: tuple[int, int]) -> None:
        """Clear, then walk the scene and draw every node that has a mesh attached."""
        self._require_program()
        width, height = viewport_size
        if width <= 0 or height <= 0:
            return
        glViewport(0, 0, width, height)
        glClearColor(0.08, 0.09, 0.10, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        view = camera.view_matrix()
        proj = camera.projection_matrix(aspect=width / height)
        self._draw_scene_nodes(scene, view, proj)

    def _draw_scene_nodes(self, scene: Scene, view: Mat4, proj: Mat4) -> None:
        for node in scene.root.traverse():
            mesh_ids = self._node_to_mesh.get(id(node))
            if not mesh_ids:
                continue
            skin = self._node_to_skin.get(id(node))
            for mesh_id in mesh_ids:
                gl_mesh = self._meshes[mesh_id]
                if gl_mesh.mmd_material is not None:
                    self._draw_mmd(node, mesh_id, skin, view, proj)
                elif skin is not None and self._skin_program is not None:
                    self._draw_skinned(mesh_id, skin, view, proj)
                else:
                    self._draw_unskinned(node, mesh_id, view, proj)

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
        """Inverted-hull outline: front-face culled, expanded along normals."""
        program = self._outline_skin_program if skin is not None else self._outline_program
        if program is None:
            return
        glEnable(GL_CULL_FACE)
        glCullFace(GL_FRONT)
        try:
            with use_program(program.program_id):
                self._set_camera_uniforms(program, view, proj)
                self._set_geometry_uniforms(program, node, skin)
                glUniform1f(program.uniform_location(U_EDGE_SIZE), material.edge_size)
                glUniform4fv(
                    program.uniform_location(U_EDGE_COLOR),
                    1, np.asarray(material.edge_color, dtype=np.float32),
                )
                with bind_vao(gl_mesh.vao):
                    glDrawElements(
                        GL_TRIANGLES, gl_mesh.index_count,
                        GL_UNSIGNED_INT, ctypes.c_void_p(0),
                    )
        finally:
            glDisable(GL_CULL_FACE)

    def _draw_toon(
        self, node: Node, gl_mesh: GLMesh, skin: Skin | None, material: MMDMaterial,
        view: Mat4, proj: Mat4,
    ) -> None:
        """Toon-shaded forward pass — Lambert × ramp + sphere composite + Blinn-Phong."""
        program = self._toon_skin_program if skin is not None else self._toon_program
        if program is None:
            return
        if not material.is_double_sided:
            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)
        try:
            with use_program(program.program_id):
                self._set_camera_uniforms(program, view, proj)
                self._set_geometry_uniforms(program, node, skin)
                self._bind_toon_textures(program, gl_mesh, material)
                self._bind_toon_scalars(program, material)
                with bind_vao(gl_mesh.vao):
                    glDrawElements(
                        GL_TRIANGLES, gl_mesh.index_count,
                        GL_UNSIGNED_INT, ctypes.c_void_p(0),
                    )
        finally:
            glDisable(GL_CULL_FACE)

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
        bone_matrices = _compute_bone_matrices(skin)
        glUniformMatrix4fv(
            program.uniform_location(U_BONE_MATRICES),
            bone_matrices.shape[0], GL_TRUE,
            np.ascontiguousarray(bone_matrices, dtype=np.float32),
        )

    def _bind_toon_textures(
        self, program: Program, gl_mesh: GLMesh, material: MMDMaterial,
    ) -> None:
        """Bind the three toon-pass textures (albedo / sphere / toon ramp).

        Each unit always has a real texture bound; uniform fall-throughs go
        to the white fallback so unbound samplers do not produce undefined
        sampler values that would surface as driver-specific noise.
        """
        fallback_id = self._white_fallback.texture_id if self._white_fallback else 0
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, gl_mesh.base_color_texture_id or fallback_id)
        glUniform1i(program.uniform_location(U_BASE_COLOR_TEX), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, gl_mesh.sphere_texture_id or fallback_id)
        glUniform1i(program.uniform_location(U_SPHERE_TEX), 1)

        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, gl_mesh.toon_texture_id or fallback_id)
        glUniform1i(program.uniform_location(U_TOON_TEX), 2)

        glUniform1i(
            program.uniform_location(U_SPHERE_MODE),
            int(material.sphere_mode),
        )

    def _bind_toon_scalars(self, program: Program, material: MMDMaterial) -> None:
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
            np.asarray(self._light_direction, dtype=np.float32),
        )
        glUniform3fv(
            program.uniform_location(U_LIGHT_COLOR), 1,
            np.asarray(self._light_color, dtype=np.float32),
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
        bone_matrices = _compute_bone_matrices(skin)
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
        loc_view = program.uniform_location(U_VIEW_MATRIX)
        loc_proj = program.uniform_location(U_PROJECTION_MATRIX)
        glUniformMatrix4fv(loc_view, 1, GL_TRUE, np.ascontiguousarray(view, dtype=np.float32))
        glUniformMatrix4fv(loc_proj, 1, GL_TRUE, np.ascontiguousarray(proj, dtype=np.float32))

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
    out = np.empty((joint_count, 4, 4), dtype=np.float32)
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

    for i in range(joint_count):
        joint = joints[i]
        if not isinstance(joint, Node):
            out[i] = np.eye(4, dtype=np.float32)
            continue
        out[i] = world(joint) @ inverse_binds[i]
    return out
