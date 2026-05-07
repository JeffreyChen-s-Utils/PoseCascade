#version 410 core
//
// pass: effect / fullscreen
//
// Standard quad-cover vertex shader used by every post-effect pass.
// Drawing six vertices in NDC and computing UVs on the fly avoids
// shipping a tiny VBO with explicit corner coordinates.

const vec2 POSITIONS[6] = vec2[](
    vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
    vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0)
);

out vec2 v_uv;

void main() {
    vec2 ndc = POSITIONS[gl_VertexID];
    v_uv = ndc * 0.5 + 0.5;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
