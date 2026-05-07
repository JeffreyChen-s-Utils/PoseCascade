"""Effect chain + library.

:class:`EffectChain` is the per-scene ordered list the user composes
in the UI. Each entry pairs a :class:`EffectDescriptor` with its
runtime state (``enabled`` plus a dict of uniform overrides keyed by
uniform name).

:class:`EffectLibrary` is a registry of *available* descriptors —
the four built-ins ship pre-registered, and the chain UI lets the
user pick from this library when adding a new pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from posecascade.render.effects.descriptor import (
    EffectDescriptor,
    UniformValue,
)


@dataclass
class ChainEntry:
    """One slot in an :class:`EffectChain`.

    ``uniform_overrides`` only stores *non-default* values; missing
    entries fall back to the descriptor's per-uniform ``default``. This
    keeps a serialised chain compact when the user accepts every
    default.
    """

    descriptor: EffectDescriptor
    enabled: bool = True
    uniform_overrides: dict[str, UniformValue] = field(default_factory=dict)

    def effective_value(self, uniform_name: str) -> UniformValue | None:
        """Return the user-edited value or the descriptor's default."""
        if uniform_name in self.uniform_overrides:
            return self.uniform_overrides[uniform_name]
        for uniform in self.descriptor.uniforms:
            if uniform.name == uniform_name:
                return uniform.default
        return None


@dataclass
class EffectChain:
    """Ordered list of effect-pass slots."""

    entries: list[ChainEntry] = field(default_factory=list)

    def __iter__(self):                              # noqa: ANN204 — iterates ChainEntry
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def append(self, descriptor: EffectDescriptor) -> ChainEntry:
        """Add a descriptor at the tail of the chain (enabled by default)."""
        entry = ChainEntry(descriptor=descriptor)
        self.entries.append(entry)
        return entry

    def insert(self, index: int, descriptor: EffectDescriptor) -> ChainEntry:
        entry = ChainEntry(descriptor=descriptor)
        self.entries.insert(max(0, min(index, len(self.entries))), entry)
        return entry

    def remove_at(self, index: int) -> ChainEntry | None:
        if not 0 <= index < len(self.entries):
            return None
        return self.entries.pop(index)

    def move(self, from_index: int, to_index: int) -> None:
        """Move entry from ``from_index`` to ``to_index`` (clamped)."""
        if not 0 <= from_index < len(self.entries):
            return
        entry = self.entries.pop(from_index)
        clamped = max(0, min(to_index, len(self.entries)))
        self.entries.insert(clamped, entry)

    def set_enabled(self, index: int, enabled: bool) -> None:
        if 0 <= index < len(self.entries):
            self.entries[index].enabled = bool(enabled)

    def set_uniform(
        self, index: int, uniform_name: str, value: UniformValue,
    ) -> None:
        if 0 <= index < len(self.entries):
            self.entries[index].uniform_overrides[uniform_name] = value

    def reset_uniform(self, index: int, uniform_name: str) -> None:
        """Drop the override and re-use the descriptor's default."""
        if not 0 <= index < len(self.entries):
            return
        self.entries[index].uniform_overrides.pop(uniform_name, None)


@dataclass
class EffectLibrary:
    """Registry of descriptors the UI offers in its "Add effect" menu."""

    descriptors: dict[str, EffectDescriptor] = field(default_factory=dict)

    def register(self, descriptor: EffectDescriptor) -> EffectDescriptor:
        """Add or replace a descriptor; the canonical key is its ``name``."""
        self.descriptors[descriptor.name] = descriptor
        return descriptor

    def find(self, name: str) -> EffectDescriptor | None:
        return self.descriptors.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.descriptors.keys()))
