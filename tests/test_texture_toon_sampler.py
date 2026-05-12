"""Tests for the toon-ramp sampler tuning helper.

:func:`posecascade.gl.texture.set_toon_sampler_params` rebinds a GL texture
and re-applies ``GL_NEAREST`` + ``GL_CLAMP_TO_EDGE`` so the MMD toon ramp
gives crisp banded shading instead of being interpolated into a gradient.
This test exercises the helper with the OpenGL bindings replaced so it
runs without needing a real GL context.
"""
from __future__ import annotations

import pytest


def test_set_toon_sampler_params_writes_nearest_and_clamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The helper sets MIN/MAG filter to NEAREST and WRAP_S/T to CLAMP_TO_EDGE."""
    from posecascade.gl import texture as tex  # noqa: PLC0415

    parameter_calls: list[tuple[int, int, int]] = []
    bind_calls: list[tuple[int, int]] = []

    def fake_tex_parameteri(target: int, pname: int, value: int) -> None:
        parameter_calls.append((target, pname, value))

    def fake_bind_texture(target: int, texture_id: int) -> None:
        bind_calls.append((target, texture_id))

    monkeypatch.setattr(tex, "glTexParameteri", fake_tex_parameteri)
    monkeypatch.setattr(tex, "glBindTexture", fake_bind_texture)

    tex.set_toon_sampler_params(texture_id=42)

    # Two binds: select target tex, then unbind.
    assert bind_calls[0] == (tex.GL_TEXTURE_2D, 42)
    assert bind_calls[-1] == (tex.GL_TEXTURE_2D, 0)

    settings = {(pname, value) for (_t, pname, value) in parameter_calls}
    assert (tex.GL_TEXTURE_MIN_FILTER, tex.GL_NEAREST) in settings
    assert (tex.GL_TEXTURE_MAG_FILTER, tex.GL_NEAREST) in settings
    assert (tex.GL_TEXTURE_WRAP_S, tex.GL_CLAMP_TO_EDGE) in settings
    assert (tex.GL_TEXTURE_WRAP_T, tex.GL_CLAMP_TO_EDGE) in settings


def test_set_toon_sampler_params_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling the helper twice on the same id is safe (no error, same params)."""
    from posecascade.gl import texture as tex  # noqa: PLC0415

    monkeypatch.setattr(tex, "glTexParameteri", lambda *a, **kw: None)
    monkeypatch.setattr(tex, "glBindTexture", lambda *a, **kw: None)
    tex.set_toon_sampler_params(texture_id=7)
    tex.set_toon_sampler_params(texture_id=7)
