"""Path-traversal-safe resolution for asset references inside model files.

Every glTF ``uri``, OBJ ``mtllib``, FBX embedded reference, etc. MUST be
resolved through :func:`resolve_safe`. Direct ``Path.resolve()`` calls on
user-controlled strings are forbidden in new code.
"""
from __future__ import annotations

from pathlib import Path

from posecascade.errors import UnsafePathError


def resolve_safe(root: Path, reference: str) -> Path:
    """Resolve ``reference`` relative to ``root`` and assert it stays inside ``root``.

    Rejects:
    - empty references
    - absolute paths (``/etc/passwd``, ``C:\\Windows\\system32``)
    - parent traversal (``../../etc/passwd``)
    - resolved paths that escape ``root`` (covers symlink attacks, since
      :py:meth:`Path.resolve` follows symlinks).

    The target file is NOT required to exist — asset references are validated
    before the file is opened.
    """
    if not reference:
        raise UnsafePathError("empty asset reference")
    candidate = Path(reference)
    if candidate.is_absolute():
        raise UnsafePathError(f"absolute path not allowed: {reference!r}")
    try:
        resolved_root = root.resolve(strict=True)
    except FileNotFoundError as err:
        raise UnsafePathError(f"asset root does not exist: {root!r}") from err
    resolved = (resolved_root / candidate).resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as err:
        raise UnsafePathError(f"path escapes asset root: {reference!r}") from err
    return resolved
