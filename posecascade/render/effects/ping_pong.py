"""GL ping-pong FBO pool for the post-effect chain.

Layered design so the swap logic stays unit-testable:

- :class:`PingPongState` is pure data: tracks which texture the next
  pass should write to, which the next pass should read from for each
  named source, and a sampler-unit cursor for input bindings. No GL.
- :class:`EffectPingPong` is the thin GL adapter — it owns two FBOs +
  two colour textures + a depth attachment, and provides the trio of
  ``bind_input`` / ``bind_output`` / ``draw_quad`` callbacks the
  :class:`EffectChainExecutor` consumes. Plus ``before_pass`` for the
  ``glUseProgram`` switch the executor delegates.

The chain reads the renderer's main scene texture as ``main_color``
on the first pass, then each pass's ``output_name`` (typically
``"result"``) becomes the source for the next. After the chain
completes :meth:`EffectPingPong.present` blits the latest output
texture to the default framebuffer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from posecascade.errors import GLError

# Sampler units 0..7 are reserved by the toon pass + scene-side bindings;
# effects start their own sampler counter at unit 8 to avoid stomping
# on the renderer's per-frame texture state when the chain is dispatched
# right after the toon draw.
EFFECT_SAMPLER_UNIT_BASE = 8
_FULLSCREEN_VERTEX_COUNT = 6
_DEFAULT_FRAMEBUFFER = 0
_MAX_TEXTURE_UNITS = 32


@dataclass
class PingPongState:
    """Pure swap-state model for the executor's bind / draw callbacks.

    The executor calls ``bind_output`` first; we record the next write
    slot and treat the texture that *was* in that slot as the new
    binding for the descriptor's ``output_name``. Subsequent passes can
    therefore read ``"result"`` and pick up the texture the previous
    pass wrote — the standard "previous output" chaining contract.
    """

    sources: dict[str, int] = field(default_factory=dict)
    color_textures: tuple[int, int] = (0, 0)
    framebuffers: tuple[int, int] = (0, 0)
    next_write_index: int = 0
    sampler_cursor: int = EFFECT_SAMPLER_UNIT_BASE

    def begin_chain(self, *, main_color_texture: int) -> None:
        """Reset state at the start of a frame's chain."""
        self.sources = {"main_color": main_color_texture}
        self.next_write_index = 0
        self.sampler_cursor = EFFECT_SAMPLER_UNIT_BASE

    def begin_pass(self) -> None:
        """Reset the sampler-unit cursor between passes."""
        self.sampler_cursor = EFFECT_SAMPLER_UNIT_BASE

    def take_write_target(self, output_name: str) -> tuple[int, int]:
        """Return the ``(framebuffer_id, texture_id)`` for the next pass.

        Records the texture under ``output_name`` so a later pass that
        binds ``output_name`` as input picks up this pass's output.
        Advances the ping-pong index.
        """
        index = self.next_write_index
        framebuffer = self.framebuffers[index]
        texture = self.color_textures[index]
        self.sources[output_name] = texture
        self.next_write_index = 1 - index
        return framebuffer, texture

    def take_input_texture(self, source: str) -> int:
        """Return the texture ID registered for the named source.

        Falls back to ``0`` when the source is unknown — drivers treat
        a binding of texture 0 as "white", which matches what the toon
        pass already does for missing material textures.
        """
        return self.sources.get(source, 0)

    def take_sampler_unit(self) -> int:
        """Allocate the next sampler unit for this pass's input."""
        unit = self.sampler_cursor
        if unit >= _MAX_TEXTURE_UNITS:
            raise GLError(
                f"effect chain ran out of sampler units (cap {_MAX_TEXTURE_UNITS})",
            )
        self.sampler_cursor += 1
        return unit

    def latest_output_texture(self) -> int:
        """Return the most recently written texture, or ``0`` if no pass ran."""
        return self.sources.get("result", self.sources.get("main_color", 0))


@dataclass
class EffectPingPong:
    """GL-bound adapter exposing the executor's three callbacks.

    Construction is empty so the renderer can build it before its GL
    context is ready. :meth:`allocate` does the actual texture / FBO
    creation; :meth:`deallocate` releases everything.
    """

    width: int = 0
    height: int = 0
    state: PingPongState = field(default_factory=PingPongState)
    _depth_renderbuffer: int = 0
    _empty_vao: int = 0
    _allocated: bool = False
    # Most-recently-bound program — kept so ``bind_input`` can pull the
    # uniform location off the right program without the executor having
    # to thread it through every callback.
    _current_program_id: int = 0

    def allocate(self, width: int, height: int) -> None:
        """Create / resize the ping-pong FBOs to ``width × height``."""
        from OpenGL.GL import (  # noqa: PLC0415
            GL_COLOR_ATTACHMENT0,
            GL_DEPTH24_STENCIL8,
            GL_DEPTH_STENCIL_ATTACHMENT,
            GL_FRAMEBUFFER,
            GL_FRAMEBUFFER_COMPLETE,
            GL_LINEAR,
            GL_RENDERBUFFER,
            GL_RGBA,
            GL_RGBA8,
            GL_TEXTURE_2D,
            GL_TEXTURE_MAG_FILTER,
            GL_TEXTURE_MIN_FILTER,
            GL_UNSIGNED_BYTE,
            glBindFramebuffer,
            glBindRenderbuffer,
            glBindTexture,
            glCheckFramebufferStatus,
            glFramebufferRenderbuffer,
            glFramebufferTexture2D,
            glGenFramebuffers,
            glGenRenderbuffers,
            glGenTextures,
            glGenVertexArrays,
            glRenderbufferStorage,
            glTexImage2D,
            glTexParameteri,
        )
        if self._allocated:
            self.deallocate()
        textures = [int(handle) for handle in glGenTextures(2)]
        framebuffers = [int(handle) for handle in glGenFramebuffers(2)]
        depth = int(glGenRenderbuffers(1))
        glBindRenderbuffer(GL_RENDERBUFFER, depth)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height)
        for i in range(2):
            glBindTexture(GL_TEXTURE_2D, textures[i])
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                GL_RGBA, GL_UNSIGNED_BYTE, None,
            )
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glBindFramebuffer(GL_FRAMEBUFFER, framebuffers[i])
            glFramebufferTexture2D(
                GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_2D, textures[i], 0,
            )
            glFramebufferRenderbuffer(
                GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                GL_RENDERBUFFER, depth,
            )
            status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
            if status != GL_FRAMEBUFFER_COMPLETE:
                raise GLError(
                    f"effect ping-pong FBO {i} incomplete (status={status})",
                )
        glBindFramebuffer(GL_FRAMEBUFFER, _DEFAULT_FRAMEBUFFER)
        empty_vao = int(glGenVertexArrays(1))
        self.width = int(width)
        self.height = int(height)
        self.state = PingPongState(
            color_textures=(textures[0], textures[1]),
            framebuffers=(framebuffers[0], framebuffers[1]),
        )
        self._depth_renderbuffer = depth
        self._empty_vao = empty_vao
        self._allocated = True

    def deallocate(self) -> None:
        """Free GL resources. Safe to call repeatedly."""
        if not self._allocated:
            return
        from OpenGL.GL import (  # noqa: PLC0415
            glDeleteFramebuffers,
            glDeleteRenderbuffers,
            glDeleteTextures,
            glDeleteVertexArrays,
        )
        glDeleteTextures(2, list(self.state.color_textures))
        glDeleteFramebuffers(2, list(self.state.framebuffers))
        glDeleteRenderbuffers(1, [self._depth_renderbuffer])
        glDeleteVertexArrays(1, [self._empty_vao])
        self._depth_renderbuffer = 0
        self._empty_vao = 0
        self._allocated = False
        self.state = PingPongState()

    # ----- callbacks consumed by EffectChainExecutor.run --------------
    def begin_chain(self, *, main_color_texture: int) -> None:
        """Reset for a new frame; call once per :meth:`apply_effect_chain`."""
        self.state.begin_chain(main_color_texture=main_color_texture)

    def before_pass(self, program_id: int) -> None:
        from OpenGL.GL import glUseProgram  # noqa: PLC0415

        glUseProgram(int(program_id))
        self._current_program_id = int(program_id)
        self.state.begin_pass()

    def bind_input(self, sampler_name: str, source: str) -> None:
        from OpenGL.GL import (  # noqa: PLC0415
            GL_TEXTURE0,
            GL_TEXTURE_2D,
            glActiveTexture,
            glBindTexture,
            glGetUniformLocation,
            glUniform1i,
        )
        unit = self.state.take_sampler_unit()
        texture = self.state.take_input_texture(source)
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(GL_TEXTURE_2D, int(texture))
        location = int(glGetUniformLocation(self._current_program_id, sampler_name))
        if location >= 0:
            glUniform1i(location, unit)

    def bind_output(self, output_name: str) -> None:
        from OpenGL.GL import (  # noqa: PLC0415
            GL_COLOR_BUFFER_BIT,
            GL_DEPTH_BUFFER_BIT,
            GL_FRAMEBUFFER,
            glBindFramebuffer,
            glClear,
            glClearColor,
            glViewport,
        )
        framebuffer, _texture = self.state.take_write_target(output_name)
        glBindFramebuffer(GL_FRAMEBUFFER, int(framebuffer))
        glViewport(0, 0, self.width, self.height)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def draw_quad(self) -> None:
        from OpenGL.GL import (  # noqa: PLC0415
            GL_TRIANGLES,
            glBindVertexArray,
            glDrawArrays,
        )
        glBindVertexArray(self._empty_vao)
        glDrawArrays(GL_TRIANGLES, 0, _FULLSCREEN_VERTEX_COUNT)
        glBindVertexArray(0)

    def present(self, *, default_framebuffer: int = _DEFAULT_FRAMEBUFFER) -> None:
        """Blit the latest output texture into ``default_framebuffer``.

        The blit is read-from-the-final-FBO, draw-into-the-default —
        a single ``glBlitFramebuffer`` rather than a textured quad pass
        so the post-effect output preserves its sRGB encoding under the
        driver's blit semantics rather than going through another
        sampling step.
        """
        if not self._allocated:
            return
        from OpenGL.GL import (  # noqa: PLC0415
            GL_COLOR_BUFFER_BIT,
            GL_DRAW_FRAMEBUFFER,
            GL_LINEAR,
            GL_READ_FRAMEBUFFER,
            glBindFramebuffer,
            glBlitFramebuffer,
        )
        # Identify the FBO that holds the most recent output. The state
        # has already toggled ``next_write_index`` past it, so the prior
        # index is what we want.
        last_index = 1 - self.state.next_write_index
        read_fbo = int(self.state.framebuffers[last_index])
        glBindFramebuffer(GL_READ_FRAMEBUFFER, read_fbo)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, int(default_framebuffer))
        glBlitFramebuffer(
            0, 0, self.width, self.height,
            0, 0, self.width, self.height,
            GL_COLOR_BUFFER_BIT, GL_LINEAR,
        )
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, _DEFAULT_FRAMEBUFFER)
