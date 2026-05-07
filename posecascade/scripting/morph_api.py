"""Sandbox-facing facade for driving morph weights.

The renderer / morph applier consumes a ``{name: weight}`` map each
frame; this class is the script-side surface that lets either a
sandboxed Python script or the declarative-animation runtime push
weights into that map without reaching into engine internals.

Construction takes a target dict (or a setter callable) — the engine
provides whichever shape it has at hand. For tests / standalone use,
:class:`MorphApi` defaults to its own internal dict so the caller
can read back the latest weights via :meth:`current_weights`.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping


class MorphApi:
    """Curated morph-weight surface for sandboxed user scripts."""

    def __init__(
        self,
        target: dict[str, float] | None = None,
        *,
        setter: Callable[[str, float], None] | None = None,
    ) -> None:
        self._weights: dict[str, float] = target if target is not None else {}
        self._setter = setter

    def set(self, name: str, weight: float) -> None:
        """Set ``name``'s weight for this frame (0..1 typical, unclamped)."""
        if self._setter is not None:
            self._setter(str(name), float(weight))
        self._weights[str(name)] = float(weight)

    def clear(self) -> None:
        """Drop every weight — call at the start of each frame if the caller
        owns the dict and wants clean per-frame semantics rather than the
        accumulating-until-overwritten default."""
        self._weights.clear()

    def current_weights(self) -> Mapping[str, float]:
        """Snapshot of the current weight map. Read-only-by-convention."""
        return self._weights

    def update(self, weights: Mapping[str, float]) -> None:
        """Apply a batch of weights. Keys not present in ``weights`` are NOT
        cleared — call :meth:`clear` first if the caller wants to drop
        previous-frame entries that no longer have a curve."""
        for name, weight in weights.items():
            self.set(name, weight)


__all__ = ["MorphApi"]
