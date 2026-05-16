#version 410 core
//
// pass: ground / checker
// attributes (location):
//   0 a_position (vec3)   — XZ plane vertex, Y = 0
// uniforms:
//   u_modelMatrix, u_viewMatrix, u_projMatrix
//   u_normalMatrix (mat3) — kept for API parity with the toon vert shader
//
// Passes world-space XZ through to the frag shader so the checker pattern
// stays anchored to world coords rather than scrolling with the camera.

layout(location = 0) in vec3 a_position;

uniform mat4 u_modelMatrix;
uniform mat4 u_viewMatrix;
uniform mat4 u_projMatrix;

out vec2 v_world_xz;

void main() {
    vec4 world = u_modelMatrix * vec4(a_position, 1.0);
    v_world_xz = world.xz;
    gl_Position = u_projMatrix * u_viewMatrix * world;
}
