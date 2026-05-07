"""GL-backed runner for an :class:`EffectChain`.

The executor is the missing link between the descriptor data layer and
visible post-effect output:

- :meth:`compile_chain` walks every descriptor in the chain, reads its
  fragment shader from disk, and compiles a :class:`Program` paired
  with the shared ``shaders/effects/fullscreen.vert``. Programs are
  cached so re-running the same chain doesn't recompile.
- :meth:`run` walks the enabled entries, for each one binds the next
  output FBO + the input textures the descriptor declared, sets every
  uniform from the entry's ``effective_value``, and draws a six-vertex
  fullscreen quad.

Two FBOs are allocated as a ping-pong pair so each pass reads from the
*previous* pass's output. Constructing the executor + calling
:meth:`compile_chain` requires a current GL context; :meth:`run`
requires that the renderer has finished its main pass and bound the
result as ``main_color``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from posecascade.errors import GLError
from posecascade.gl.shader import Program, compile_program
from posecascade.render.effects.chain import ChainEntry, EffectChain
from posecascade.render.effects.descriptor import (
    EffectDescriptor,
    EffectUniformKind,
)

DEFAULT_FULLSCREEN_VERT = "shaders/effects/fullscreen.vert"
_FULLSCREEN_VERTEX_COUNT = 6


@dataclass
class CompiledEffect:
    """One descriptor's compiled GL state — program + uniform-name cache."""

    descriptor: EffectDescriptor
    program: Program


@dataclass
class EffectChainExecutor:
    """Compiles + drives an :class:`EffectChain` end-to-end on the GL side.

    Construction is deliberately empty so a renderer can build it before
    it has a chain assigned (e.g. at startup, while the user hasn't yet
    loaded a project). :meth:`compile_chain` populates the program
    cache; :meth:`run` is the per-frame entry point.
    """

    project_root: Path
    fullscreen_vert_path: str = DEFAULT_FULLSCREEN_VERT
    _programs: dict[str, CompiledEffect] = field(default_factory=dict, init=False)

    def compile_chain(self, chain: EffectChain) -> None:
        """Compile (and cache) every descriptor referenced by ``chain``."""
        for entry in chain.entries:
            self._compile_descriptor(entry.descriptor)

    def compiled(self, name: str) -> CompiledEffect | None:
        """Return the cached :class:`CompiledEffect` for ``name`` (or ``None``)."""
        return self._programs.get(name)

    def run(
        self,
        chain: EffectChain,
        *,
        bind_input: callable,                                      # noqa: ANN001
        bind_output: callable,                                     # noqa: ANN001
        draw_quad: callable,                                       # noqa: ANN001
        before_pass: callable | None = None,                       # noqa: ANN001
    ) -> None:
        """Walk each enabled entry + execute its pass through the supplied hooks.

        ``bind_input(sampler_name, source_name)`` binds the texture
        sourced from ``source_name`` (e.g. ``main_color``) to the
        current program at the named uniform. ``bind_output(name)``
        switches the active framebuffer to where this pass's output
        should land. ``draw_quad()`` issues the six-vertex draw call.

        Optional ``before_pass(program_id)`` fires once per pass before
        any input / output is bound and before uniforms upload. The GL
        ping-pong wrapper hooks this to call ``glUseProgram``; tests
        that drive the executor with a fake program leave it out.

        The executor passes responsibility for FBO layout out to the
        caller because the renderer owns the FBO pool and the swap
        strategy (ping-pong is the default but a "result" buffer the
        viewport blits to is also legitimate).
        """
        for entry in chain.entries:
            if not entry.enabled:
                continue
            compiled = self._programs.get(entry.descriptor.name)
            if compiled is None:
                continue
            if before_pass is not None:
                before_pass(compiled.program.program_id)
            bind_output(entry.descriptor.output_name)
            for input_binding in entry.descriptor.inputs:
                bind_input(input_binding.sampler_name, input_binding.source)
            self._upload_uniforms(compiled, entry)
            draw_quad()

    # ----- internal ----------------------------------------------------
    def _compile_descriptor(self, descriptor: EffectDescriptor) -> None:
        if descriptor.name in self._programs:
            return
        vert_path = self.project_root / self.fullscreen_vert_path
        frag_path = self.project_root / descriptor.fragment_shader
        try:
            vert_source = vert_path.read_text(encoding="utf-8")
            frag_source = frag_path.read_text(encoding="utf-8")
        except OSError as err:
            raise GLError(
                f"effect {descriptor.name!r}: cannot read shader at "
                f"{frag_path}: {err}",
            ) from err
        program = compile_program(vert_source, frag_source)
        self._programs[descriptor.name] = CompiledEffect(
            descriptor=descriptor, program=program,
        )

    def _upload_uniforms(self, compiled: CompiledEffect, entry: ChainEntry) -> None:
        """Walk the descriptor's uniform schema + push each effective value."""
        for uniform in compiled.descriptor.uniforms:
            value = entry.effective_value(uniform.name)
            if value is None:
                continue
            self._upload_one_uniform(compiled.program, uniform.name, uniform.kind, value)

    def _upload_one_uniform(
        self,
        program: Program,
        name: str,
        kind: EffectUniformKind,
        value: object,
    ) -> None:
        # Split the GL imports off the module-load path so headless
        # tests that don't touch GL can still import this module.
        from OpenGL.GL import (  # noqa: PLC0415
            glUniform1f,
            glUniform1i,
            glUniform3fv,
            glUniform4fv,
        )
        location = program.uniform_location(name)
        if location < 0:
            return
        if kind == EffectUniformKind.SCALAR:
            glUniform1f(location, float(value))
        elif kind in (EffectUniformKind.BOOL, EffectUniformKind.INT_ENUM):
            glUniform1i(location, int(value))
        elif kind == EffectUniformKind.VEC3_COLOR:
            glUniform3fv(location, 1, _vec_floats(value, 3))
        elif kind == EffectUniformKind.VEC4_COLOR:
            glUniform4fv(location, 1, _vec_floats(value, 4))


def _vec_floats(value: object, length: int) -> list[float]:
    if not isinstance(value, list | tuple) or len(value) < length:
        return [0.0] * length
    return [float(v) for v in value[:length]]
