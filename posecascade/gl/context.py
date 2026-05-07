"""GL context wrapper.

Thin façade over the active :class:`PySide6.QtGui.QOpenGLContext`. The wrapper
exists to give the rest of the engine a single import surface and to enforce
the "all GL calls happen on the owning thread" rule.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from posecascade.errors import GLError


@dataclass
class GLContext:
    """Reference to the active GL context plus the owning thread id."""

    owner_thread_id: int
    profile: str = "core"
    version: tuple[int, int] = (4, 1)

    def assert_owned(self) -> None:
        """Raise :class:`GLError` if called from a thread that does not own this context."""
        if threading.get_ident() != self.owner_thread_id:
            raise GLError("GL call from non-owner thread; route via the render queue")


def claim_current_thread(profile: str = "core", version: tuple[int, int] = (4, 1)) -> GLContext:
    """Mark the current thread as the GL-owning thread and return the context handle."""
    return GLContext(owner_thread_id=threading.get_ident(), profile=profile, version=version)
