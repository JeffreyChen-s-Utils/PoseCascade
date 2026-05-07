#version 410 core
//
// pass: toon / outline (shared)
// uniforms:
//   u_edgeColor (vec4) — per-material outline RGBA

uniform vec4 u_edgeColor;

out vec4 frag_color;

void main() {
    frag_color = u_edgeColor;
}
