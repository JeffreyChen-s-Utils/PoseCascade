"""Real bind / unbind context managers backed by PyOpenGL.

Each helper saves the prior binding (zero outside a ``with`` block) and
restores it on exit, so nested binds do not leak state across siblings.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from OpenGL.GL import GL_DRAW_FRAMEBUFFER as _GL_DRAW_FRAMEBUFFER
from OpenGL.GL import (
    GL_DRAW_FRAMEBUFFER_BINDING,
    GL_VERTEX_ARRAY_BINDING,
    glBindFramebuffer,
    glBindVertexArray,
    glGetIntegerv,
    glUseProgram,
)

_FBO_TARGET = _GL_DRAW_FRAMEBUFFER


@contextmanager
def bind_vao(vao_id: int) -> Iterator[int]:
    """Bind a VAO for the duration of the ``with`` block; restore prior on exit."""
    previous = int(glGetIntegerv(GL_VERTEX_ARRAY_BINDING))
    glBindVertexArray(int(vao_id))
    try:
        yield vao_id
    finally:
        glBindVertexArray(previous)


@contextmanager
def bind_framebuffer(fbo_id: int) -> Iterator[int]:
    """Bind a draw framebuffer for the duration of the ``with`` block."""
    previous = int(glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING))
    glBindFramebuffer(_FBO_TARGET, int(fbo_id))
    try:
        yield fbo_id
    finally:
        glBindFramebuffer(_FBO_TARGET, previous)


@contextmanager
def use_program(program_id: int) -> Iterator[int]:
    """Bind a shader program for the duration of the ``with`` block."""
    glUseProgram(int(program_id))
    try:
        yield program_id
    finally:
        glUseProgram(0)
