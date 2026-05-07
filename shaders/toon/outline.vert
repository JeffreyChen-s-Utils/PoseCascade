#version 410 core
//
// pass: toon / outline (non-skinned inverted hull)
// attributes (location):
//   0 a_position (vec3)
//   1 a_normal   (vec3)
// uniforms:
//   u_modelMatrix, u_viewMatrix, u_projMatrix, u_normalMatrix
//   u_edgeSize (float) — per-material outline thickness, in mesh-local units
//
// Pushes each vertex outwards along its mesh-local normal and renders with
// front-face culling so only the back faces remain — the classic
// inverted-hull outline. Width is in object space so distant models don't
// look uniformly thick the way a screen-space outline would.

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;

uniform mat4 u_modelMatrix;
uniform mat4 u_viewMatrix;
uniform mat4 u_projMatrix;
uniform mat3 u_normalMatrix;
uniform float u_edgeSize;

void main() {
    vec3 expanded = a_position + normalize(a_normal) * u_edgeSize;
    gl_Position = u_projMatrix * u_viewMatrix * u_modelMatrix * vec4(expanded, 1.0);
}
