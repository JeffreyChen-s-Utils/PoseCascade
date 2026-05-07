"""Tests for :class:`EffectChainExecutor` (data plumbing only — no GL)."""
from __future__ import annotations

from pathlib import Path

import pytest

from posecascade.errors import GLError
from posecascade.render.effects.builtins import register_builtins
from posecascade.render.effects.chain import EffectChain, EffectLibrary
from posecascade.render.effects.descriptor import (
    EffectDescriptor,
    EffectInput,
    EffectUniform,
    EffectUniformKind,
)
from posecascade.render.effects.executor import EffectChainExecutor

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_compile_chain_raises_when_fragment_missing(tmp_path: Path) -> None:
    """A descriptor pointing at a non-existent fragment shader raises a
    structured :class:`GLError` rather than letting the OSError bubble up."""
    descriptor = EffectDescriptor(
        name="ghost",
        fragment_shader="does/not/exist.frag",
    )
    chain = EffectChain()
    chain.append(descriptor)
    executor = EffectChainExecutor(project_root=tmp_path)
    with pytest.raises(GLError, match="cannot read shader"):
        executor.compile_chain(chain)


def test_executor_runs_each_enabled_descriptor_in_order() -> None:
    """The ``run`` hook callbacks fire once per enabled descriptor, in
    insertion order, and are skipped for disabled entries."""
    library = register_builtins(EffectLibrary())
    chain = EffectChain()
    chain.append(library.find("autoluminous"))
    chain.append(library.find("o_greener"))
    chain.set_enabled(1, False)
    executor = EffectChainExecutor(project_root=_PROJECT_ROOT)
    # Pre-populate the program cache so ``run`` doesn't try to compile.
    from posecascade.render.effects.executor import CompiledEffect  # noqa: PLC0415
    for entry in chain.entries:
        executor._programs[entry.descriptor.name] = CompiledEffect(           # noqa: SLF001
            descriptor=entry.descriptor,
            program=type("FakeProgram", (), {
                "program_id": 0,
                "uniform_location": lambda self, name: -1,
            })(),
        )
    output_calls: list[str] = []
    input_calls: list[tuple[str, str]] = []
    draw_calls: list[None] = []

    def bind_input(sampler: str, source: str) -> None:
        input_calls.append((sampler, source))

    def bind_output(name: str) -> None:
        output_calls.append(name)

    def draw_quad() -> None:
        draw_calls.append(None)

    executor.run(
        chain, bind_input=bind_input, bind_output=bind_output, draw_quad=draw_quad,
    )
    # Only the enabled "autoluminous" entry fires.
    assert output_calls == ["result"]
    assert ("u_main_color", "main_color") in input_calls
    assert len(draw_calls) == 1


def test_compiled_lookup_returns_none_for_unknown() -> None:
    executor = EffectChainExecutor(project_root=_PROJECT_ROOT)
    assert executor.compiled("unknown") is None


def test_compile_chain_idempotent(tmp_path: Path) -> None:
    """Calling ``compile_chain`` twice on the same chain doesn't
    re-compile — the program cache short-circuits the second pass."""
    descriptor = EffectDescriptor(
        name="reused",
        fragment_shader="shaders/effects/o_greener.frag",
        uniforms=(
            EffectUniform(
                name="amount",
                kind=EffectUniformKind.SCALAR,
                default=0.5,
            ),
        ),
    )
    chain = EffectChain()
    chain.append(descriptor)
    executor = EffectChainExecutor(project_root=_PROJECT_ROOT)
    # First call would actually compile (we let it raise instead — no GL
    # context). Patch the compile path to a no-op for this test so we
    # can verify cache behaviour without a real shader compile.
    from posecascade.render.effects import executor as executor_module  # noqa: PLC0415
    calls: list[str] = []
    real_compile = executor_module.compile_program
    def counting_compile(vert: str, frag: str) -> object:
        calls.append(descriptor.name)
        return type("FakeProgram", (), {
            "program_id": 0,
            "uniform_location": lambda self, _name: -1,
        })()
    executor_module.compile_program = counting_compile
    try:
        executor.compile_chain(chain)
        executor.compile_chain(chain)
    finally:
        executor_module.compile_program = real_compile
    assert calls == ["reused"]


# Keep the unused symbols load-bearing for IDE jumps.
__all__ = ["EffectInput", "register_builtins"]
