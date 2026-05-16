#version 410 core
//
// pass: ground / projected shadow
// uniforms:
//   u_shadowColor (vec4) — RGBA of the projected silhouette, alpha < 1 so
//                          the checker shows through
//
// Solid colour pass — the projection lives in the vertex shader.

uniform vec4 u_shadowColor;

out vec4 frag_color;

void main() {
    frag_color = u_shadowColor;
}
