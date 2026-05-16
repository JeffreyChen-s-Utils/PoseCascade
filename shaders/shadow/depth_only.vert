#version 410 core
//
// pass: shadow / depth-only (non-skinned)
// attributes (location):
//   0 a_position (vec3)
// uniforms:
//   u_lightSpaceMatrix (mat4) — projection * view from the light's POV
//   u_modelMatrix (mat4)
//
// Run from the light's point of view to populate the shadow depth
// texture. The fragment shader is empty; OpenGL writes ``gl_FragDepth``
// automatically from this stage's clip-space Z.

layout(location = 0) in vec3 a_position;

uniform mat4 u_lightSpaceMatrix;
uniform mat4 u_modelMatrix;

void main() {
    gl_Position = u_lightSpaceMatrix * u_modelMatrix * vec4(a_position, 1.0);
}
