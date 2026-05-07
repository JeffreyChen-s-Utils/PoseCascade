"""OpenGL context, shader, and resource helpers.

Every ``gl*`` call MUST originate from the render thread that owns the
:class:`~posecascade.gl.context.GLContext`. Cross-thread access is a bug.
"""
