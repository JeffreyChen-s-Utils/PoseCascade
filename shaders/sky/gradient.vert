#version 410 core
//
// pass: sky / gradient
//
// Fullscreen triangle (3 hard-coded vertices) so the vertex shader
// doesn't need an attached VAO — the renderer issues a 3-vertex
// drawArrays and the rasterizer hands a screen-covering primitive to
// the fragment shader. ``v_uv`` is the screen-space [0,1]² coord
// the fragment uses for the vertical gradient.

out vec2 v_uv;

void main() {
    vec2 corners[3] = vec2[3](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );
    vec2 p = corners[gl_VertexID];
    v_uv = p * 0.5 + 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
}
