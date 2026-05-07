"""Tests for the post-effect FBO ping-pong (pure swap logic)."""
from __future__ import annotations

from pathlib import Path

import pytest

from posecascade.errors import GLError
from posecascade.render.effects.builtins import register_builtins
from posecascade.render.effects.chain import EffectChain, EffectLibrary
from posecascade.render.effects.executor import CompiledEffect, EffectChainExecutor
from posecascade.render.effects.ping_pong import (
    EFFECT_SAMPLER_UNIT_BASE,
    EffectPingPong,
    PingPongState,
)


def _fake_compiled(name: str, descriptor) -> CompiledEffect:
    """Build a stand-in :class:`CompiledEffect` whose program returns -1 locations."""
    program = type(
        "FakeProgram", (), {
            "program_id": 42,
            "uniform_location": lambda self, _name: -1,
        },
    )()
    return CompiledEffect(descriptor=descriptor, program=program)


# ----- PingPongState ----------------------------------------------------
def test_state_begin_chain_seeds_main_color() -> None:
    state = PingPongState(color_textures=(11, 22), framebuffers=(33, 44))
    state.begin_chain(main_color_texture=99)
    assert state.sources == {"main_color": 99}
    assert state.next_write_index == 0
    assert state.sampler_cursor == EFFECT_SAMPLER_UNIT_BASE


def test_state_take_write_target_alternates() -> None:
    """Two passes in a row alternate which FBO + texture they target."""
    state = PingPongState(color_textures=(100, 200), framebuffers=(10, 20))
    state.begin_chain(main_color_texture=99)
    fb_a, tex_a = state.take_write_target("result")
    assert (fb_a, tex_a) == (10, 100)
    fb_b, tex_b = state.take_write_target("result")
    assert (fb_b, tex_b) == (20, 200)
    fb_c, tex_c = state.take_write_target("result")
    assert (fb_c, tex_c) == (10, 100)


def test_state_records_output_under_descriptor_name() -> None:
    """A pass's output_name becomes addressable by subsequent ``bind_input``."""
    state = PingPongState(color_textures=(100, 200), framebuffers=(10, 20))
    state.begin_chain(main_color_texture=99)
    state.take_write_target("custom_output")
    assert state.take_input_texture("custom_output") == 100
    # main_color is still reachable.
    assert state.take_input_texture("main_color") == 99


def test_state_unknown_source_falls_back_to_zero() -> None:
    state = PingPongState(color_textures=(100, 200), framebuffers=(10, 20))
    state.begin_chain(main_color_texture=99)
    assert state.take_input_texture("does_not_exist") == 0


def test_state_sampler_cursor_advances_then_resets_each_pass() -> None:
    state = PingPongState(color_textures=(100, 200), framebuffers=(10, 20))
    state.begin_chain(main_color_texture=99)
    state.begin_pass()
    assert state.take_sampler_unit() == EFFECT_SAMPLER_UNIT_BASE
    assert state.take_sampler_unit() == EFFECT_SAMPLER_UNIT_BASE + 1
    state.begin_pass()
    assert state.take_sampler_unit() == EFFECT_SAMPLER_UNIT_BASE


def test_state_overflowing_sampler_units_raises() -> None:
    state = PingPongState(color_textures=(0, 0), framebuffers=(0, 0))
    state.begin_chain(main_color_texture=0)
    state.sampler_cursor = 32
    with pytest.raises(GLError, match="sampler units"):
        state.take_sampler_unit()


def test_state_latest_output_falls_back_to_main_color() -> None:
    state = PingPongState(color_textures=(0, 0), framebuffers=(0, 0))
    state.begin_chain(main_color_texture=77)
    assert state.latest_output_texture() == 77
    state.take_write_target("result")
    assert state.latest_output_texture() == state.color_textures[0]


# ----- Driving an executor with the ping-pong's callbacks --------------
def _swap_state_only(ping_pong: EffectPingPong) -> None:
    """Hand-pre-populate the ping-pong's pure state so we can exercise it
    without GL allocation. Tests below only need the swap-logic side."""
    ping_pong.state = PingPongState(
        color_textures=(101, 202),
        framebuffers=(11, 22),
    )


def test_executor_callbacks_record_swap_through_pure_state() -> None:
    """Driving the executor with state-only callbacks reproduces the
    intended sequence of bind_output + bind_input invocations.

    We reach in and replace the GL-bound callbacks with pure-data
    stand-ins so this test runs without an OpenGL context.
    """
    library = register_builtins(EffectLibrary())
    chain = EffectChain()
    chain.append(library.find("autoluminous"))
    chain.append(library.find("o_greener"))

    project_root = Path(__file__).resolve().parent.parent
    executor = EffectChainExecutor(project_root=project_root)
    for entry in chain.entries:
        executor._programs[entry.descriptor.name] = _fake_compiled(    # noqa: SLF001
            entry.descriptor.name, entry.descriptor,
        )

    ping_pong = EffectPingPong()
    _swap_state_only(ping_pong)
    ping_pong.state.begin_chain(main_color_texture=999)
    output_calls: list[tuple[str, int, int]] = []
    input_calls: list[tuple[str, str, int, int]] = []
    pass_calls: list[int] = []

    def fake_before_pass(program_id: int) -> None:
        ping_pong.state.begin_pass()
        pass_calls.append(program_id)

    def fake_bind_output(name: str) -> None:
        framebuffer, texture = ping_pong.state.take_write_target(name)
        output_calls.append((name, framebuffer, texture))

    def fake_bind_input(sampler: str, source: str) -> None:
        unit = ping_pong.state.take_sampler_unit()
        texture = ping_pong.state.take_input_texture(source)
        input_calls.append((sampler, source, unit, texture))

    executor.run(
        chain,
        bind_input=fake_bind_input,
        bind_output=fake_bind_output,
        draw_quad=lambda: None,
        before_pass=fake_before_pass,
    )

    # Two enabled descriptors → two passes.
    assert len(pass_calls) == 2
    assert pass_calls == [42, 42]
    # Outputs alternate (FBO 0, FBO 1).
    assert [c[1] for c in output_calls] == [11, 22]
    assert [c[2] for c in output_calls] == [101, 202]
    # First pass binds main_color (texture 999, sampler unit base);
    # second pass also binds main_color since the o_greener descriptor
    # references it directly. Sampler cursor resets between passes.
    assert input_calls[0][1:] == ("main_color", EFFECT_SAMPLER_UNIT_BASE, 999)
    assert input_calls[1][1:] == ("main_color", EFFECT_SAMPLER_UNIT_BASE, 999)


def test_present_does_nothing_when_unallocated() -> None:
    """Calling ``present`` before ``allocate`` is safe — early-out."""
    ping_pong = EffectPingPong()
    ping_pong.present()  # no GL context, must not raise


def test_deallocate_is_idempotent() -> None:
    ping_pong = EffectPingPong()
    ping_pong.deallocate()  # before allocate
    ping_pong.deallocate()  # double release
