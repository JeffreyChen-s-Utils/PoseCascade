"""Tests for the post-effect descriptor + chain + TOML loader."""
from __future__ import annotations

from pathlib import Path

import pytest

from posecascade.errors import MalformedAssetError
from posecascade.render.effects.builtins import (
    builtin_descriptor_path,
    builtin_names,
    load_builtin,
    register_builtins,
)
from posecascade.render.effects.chain import (
    ChainEntry,
    EffectChain,
    EffectLibrary,
)
from posecascade.render.effects.descriptor import (
    EffectBlendMode,
    EffectDescriptor,
    EffectInput,
    EffectUniform,
    EffectUniformKind,
)
from posecascade.render.effects.loader import (
    load_chain_from_toml,
    load_descriptor_from_toml,
    serialize_chain_to_toml,
)


# ----- TOML loader ---------------------------------------------------
def test_load_descriptor_from_inline_toml() -> None:
    descriptor = load_descriptor_from_toml(
        """
        name = "tint"
        fragment_shader = "shaders/effects/tint.frag"

        [[uniforms]]
        name = "intensity"
        kind = "scalar"
        default = 0.5
        minimum = 0.0
        maximum = 1.0
        """,
    )
    assert descriptor.name == "tint"
    assert descriptor.fragment_shader == "shaders/effects/tint.frag"
    assert descriptor.uniforms[0].name == "intensity"
    assert descriptor.uniforms[0].minimum == pytest.approx(0.0)


def test_load_descriptor_missing_name_raises() -> None:
    with pytest.raises(MalformedAssetError, match="missing required"):
        load_descriptor_from_toml('fragment_shader = "x.frag"')


def test_load_descriptor_unknown_blend_mode_raises() -> None:
    with pytest.raises(MalformedAssetError, match="unknown effect blend_mode"):
        load_descriptor_from_toml(
            """
            name = "x"
            fragment_shader = "x.frag"
            blend_mode = "explosive"
            """,
        )


def test_load_descriptor_unknown_uniform_kind_raises() -> None:
    with pytest.raises(MalformedAssetError, match="unknown uniform kind"):
        load_descriptor_from_toml(
            """
            name = "x"
            fragment_shader = "x.frag"

            [[uniforms]]
            name = "u"
            kind = "weird_kind"
            """,
        )


def test_load_descriptor_invalid_toml_raises() -> None:
    with pytest.raises(MalformedAssetError, match="invalid effect TOML"):
        load_descriptor_from_toml("name =")


def test_load_descriptor_unknown_keys_are_ignored() -> None:
    """Forward-compat: a descriptor declaring future-only keys still loads."""
    descriptor = load_descriptor_from_toml(
        """
        name = "tint"
        fragment_shader = "shaders/effects/tint.frag"
        future_only = "ignore me"
        """,
    )
    assert descriptor.name == "tint"


# ----- chain operations ---------------------------------------------
def _make_descriptor(name: str = "x") -> EffectDescriptor:
    return EffectDescriptor(
        name=name,
        fragment_shader=f"{name}.frag",
        inputs=(EffectInput(sampler_name="u_main_color"),),
        uniforms=(
            EffectUniform(
                name="amount",
                kind=EffectUniformKind.SCALAR,
                default=0.5,
                minimum=0.0,
                maximum=1.0,
            ),
        ),
        blend_mode=EffectBlendMode.REPLACE,
    )


def test_chain_append_and_remove() -> None:
    chain = EffectChain()
    a = chain.append(_make_descriptor("a"))
    b = chain.append(_make_descriptor("b"))
    assert len(chain) == 2
    assert chain.entries[0] is a
    chain.remove_at(0)
    assert chain.entries == [b]


def test_chain_move_clamps_indices() -> None:
    chain = EffectChain()
    chain.append(_make_descriptor("a"))
    chain.append(_make_descriptor("b"))
    chain.append(_make_descriptor("c"))
    chain.move(0, 2)
    assert [entry.descriptor.name for entry in chain.entries] == ["b", "c", "a"]
    chain.move(0, -5)         # clamps to 0
    assert chain.entries[0].descriptor.name == "b"


def test_chain_move_out_of_range_index_no_op() -> None:
    chain = EffectChain()
    chain.append(_make_descriptor("a"))
    chain.move(5, 0)         # source index out of range
    assert chain.entries[0].descriptor.name == "a"


def test_chain_set_enabled_and_uniform_overrides() -> None:
    chain = EffectChain()
    chain.append(_make_descriptor())
    chain.set_enabled(0, False)
    assert chain.entries[0].enabled is False
    chain.set_uniform(0, "amount", 0.95)
    assert chain.entries[0].effective_value("amount") == pytest.approx(0.95)
    chain.reset_uniform(0, "amount")
    assert chain.entries[0].effective_value("amount") == pytest.approx(0.5)


def test_chain_entry_effective_value_falls_back_to_default() -> None:
    entry = ChainEntry(descriptor=_make_descriptor())
    assert entry.effective_value("amount") == pytest.approx(0.5)
    assert entry.effective_value("does_not_exist") is None


# ----- chain serialisation -------------------------------------------
def test_serialize_chain_round_trips_through_library() -> None:
    library = EffectLibrary()
    library.register(_make_descriptor("tint"))
    library.register(_make_descriptor("vignette"))
    chain = EffectChain()
    chain.append(library.find("tint"))
    chain.append(library.find("vignette"))
    chain.set_enabled(1, False)
    chain.set_uniform(0, "amount", 0.8)
    text = serialize_chain_to_toml(chain)
    reloaded = load_chain_from_toml(text, library)
    assert len(reloaded) == 2
    assert reloaded.entries[0].descriptor.name == "tint"
    assert reloaded.entries[0].effective_value("amount") == pytest.approx(0.8)
    assert reloaded.entries[1].enabled is False


def test_load_chain_skips_unknown_descriptor_names() -> None:
    library = EffectLibrary()
    library.register(_make_descriptor("known"))
    chain = load_chain_from_toml(
        """
        [[entry]]
        name = "known"
        enabled = true

        [[entry]]
        name = "missing"
        enabled = false
        """,
        library,
    )
    assert len(chain) == 1
    assert chain.entries[0].descriptor.name == "known"


# ----- built-in descriptors -----------------------------------------
def test_builtin_descriptors_load_from_disk() -> None:
    for name in builtin_names():
        path = builtin_descriptor_path(name)
        assert path.is_file(), f"missing builtin TOML: {path}"
        descriptor = load_builtin(name)
        assert descriptor.name == name


def test_builtin_descriptors_have_at_least_one_uniform() -> None:
    """Smoke-check the built-ins all expose a tunable parameter — a chain
    UI without uniforms would only offer enable / disable, defeating the
    purpose of shipping the effect at all."""
    for name in builtin_names():
        descriptor = load_builtin(name)
        assert descriptor.uniforms, f"built-in {name} has no uniforms"


def test_register_builtins_adds_each_to_library() -> None:
    library = EffectLibrary()
    register_builtins(library)
    assert set(library.names()) == set(builtin_names())


def test_builtin_fragment_paths_exist_under_shaders_dir() -> None:
    """Each built-in's ``fragment_shader`` field must point at a real file."""
    project_root = Path(__file__).resolve().parent.parent
    for name in builtin_names():
        descriptor = load_builtin(name)
        assert (project_root / descriptor.fragment_shader).is_file()


# Keep the unused symbols load-bearing for IDE jumps.
__all__ = ["EffectBlendMode", "EffectInput"]
