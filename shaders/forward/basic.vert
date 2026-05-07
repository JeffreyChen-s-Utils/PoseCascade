#version 410 core
//
// pass: forward / basic
// attributes (location):
//   0 a_position (vec3)
//   1 a_normal   (vec3)
//   3 a_texcoord_0 (vec2)
// uniforms:
//   u_modelMatrix, u_viewMatrix, u_projMatrix, u_normalMatrix

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 3) in vec2 a_texcoord_0;

uniform mat4 u_modelMatrix;
uniform mat4 u_viewMatrix;
uniform mat4 u_projMatrix;
uniform mat3 u_normalMatrix;

out vec3 v_normal;
out vec2 v_uv;

void main() {
    v_normal = normalize(u_normalMatrix * a_normal);
    v_uv = a_texcoord_0;
    gl_Position = u_projMatrix * u_viewMatrix * u_modelMatrix * vec4(a_position, 1.0);
}
