"""Shader compile and link helpers."""
from __future__ import annotations

from dataclasses import dataclass

from OpenGL.GL import (
    GL_COMPILE_STATUS,
    GL_FRAGMENT_SHADER,
    GL_LINK_STATUS,
    GL_VERTEX_SHADER,
    glAttachShader,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDeleteProgram,
    glDeleteShader,
    glGetProgramInfoLog,
    glGetProgramiv,
    glGetShaderInfoLog,
    glGetShaderiv,
    glGetUniformLocation,
    glLinkProgram,
    glShaderSource,
)

from posecascade.errors import GLError


@dataclass
class Program:
    """A linked GL shader program plus a uniform-name → location cache."""

    program_id: int
    _uniforms: dict[str, int]

    def uniform_location(self, name: str) -> int:
        cached = self._uniforms.get(name)
        if cached is not None:
            return cached
        location = int(glGetUniformLocation(self.program_id, name))
        self._uniforms[name] = location
        return location

    def delete(self) -> None:
        glDeleteProgram(self.program_id)


def compile_program(vertex_src: str, fragment_src: str) -> Program:
    """Compile + link a vertex/fragment program. Raises :class:`GLError` on failure."""
    vertex_shader = _compile_one(vertex_src, GL_VERTEX_SHADER, "vertex")
    fragment_shader = _compile_one(fragment_src, GL_FRAGMENT_SHADER, "fragment")
    program_id = int(glCreateProgram())
    glAttachShader(program_id, vertex_shader)
    glAttachShader(program_id, fragment_shader)
    glLinkProgram(program_id)
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    if not glGetProgramiv(program_id, GL_LINK_STATUS):
        log = glGetProgramInfoLog(program_id)
        glDeleteProgram(program_id)
        raise GLError(f"shader link failed: {_decode(log)}")
    return Program(program_id=program_id, _uniforms={})


def _compile_one(source: str, stage: int, label: str) -> int:
    shader_id = int(glCreateShader(stage))
    glShaderSource(shader_id, source)
    glCompileShader(shader_id)
    if not glGetShaderiv(shader_id, GL_COMPILE_STATUS):
        log = glGetShaderInfoLog(shader_id)
        glDeleteShader(shader_id)
        raise GLError(f"{label} shader compile failed: {_decode(log)}")
    return shader_id


def _decode(log: object) -> str:
    if isinstance(log, bytes):
        return log.decode("utf-8", errors="replace").strip()
    return str(log).strip()
