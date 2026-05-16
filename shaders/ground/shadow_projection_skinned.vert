#version 410 core
//
// pass: ground / projected shadow (skinned)
// attributes (location):
//   0 a_position (vec3)
//   4 a_joints_0  (uvec4)
//   5 a_weights_0 (vec4)
// uniforms:
//   u_viewMatrix, u_projMatrix
//   u_boneMatrices[MAX_BONES] — already include the model matrix
//   u_lightDirection (vec3)
//   u_groundY (float)

#define MAX_BONES 384

layout(location = 0) in vec3 a_position;
layout(location = 4) in uvec4 a_joints_0;
layout(location = 5) in vec4 a_weights_0;

uniform mat4 u_viewMatrix;
uniform mat4 u_projMatrix;
uniform mat4 u_boneMatrices[MAX_BONES];
uniform vec3 u_lightDirection;
uniform float u_groundY;

const float _MIN_LIGHT_Y = 0.05;
const float _SHADOW_Y_OFFSET = 0.001;

void main() {
    mat4 skin =
          a_weights_0.x * u_boneMatrices[a_joints_0.x]
        + a_weights_0.y * u_boneMatrices[a_joints_0.y]
        + a_weights_0.z * u_boneMatrices[a_joints_0.z]
        + a_weights_0.w * u_boneMatrices[a_joints_0.w];

    vec4 world = skin * vec4(a_position, 1.0);
    float ly = max(u_lightDirection.y, _MIN_LIGHT_Y);
    float t = (world.y - u_groundY) / ly;
    vec3 projected = vec3(
        world.x - t * u_lightDirection.x,
        u_groundY + _SHADOW_Y_OFFSET,
        world.z - t * u_lightDirection.z
    );
    gl_Position = u_projMatrix * u_viewMatrix * vec4(projected, 1.0);
}
