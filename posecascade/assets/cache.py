"""Asset cache keyed by content hash.

The cache stores CPU-side :class:`~posecascade.assets.types.Mesh` and
:class:`~posecascade.assets.types.Texture` objects plus the ids of any GPU
resources uploaded for them. Eviction and re-upload are out of scope for the
skeleton — this is the registry, not the policy engine.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def content_key(payload: bytes) -> str:
    """Compute a stable cache key from raw bytes (SHA-256, non-security use)."""
    digest = hashlib.sha256(payload, usedforsecurity=False)
    return digest.hexdigest()


@dataclass
class AssetCache[T]:
    """Generic id-keyed asset registry with simple insert/get semantics."""

    _items: dict[str, T] = field(default_factory=dict)

    def put(self, key: str, value: T) -> T:
        self._items[key] = value
        return value

    def get(self, key: str) -> T | None:
        return self._items.get(key)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._items

    def __len__(self) -> int:
        return len(self._items)
