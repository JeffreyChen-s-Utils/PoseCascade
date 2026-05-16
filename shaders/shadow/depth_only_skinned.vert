#version 410 core
//
// pass: shadow / depth-only (skinned)
// attributes (location):
//   0 a_position (vec3)
//   4 a_joints_0 (uvec4)
//   5 a_weights_0 (vec4)
// uniforms:
//   u_lightSpaceMatrix (mat4)
//   u_boneMatrices[MAX_BONES] — already include the model matrix

#define MAX_BONES 384

layout(location = 0) in vec3 a_position;
layout(location = 4) in uvec4 a_joints_0;
layout(location = 5) in vec4 a_weights_0;

uniform mat4 u_lightSpaceMatrix;
uniform mat4 u_boneMatrices[MAX_BONES];

void main() {
    mat4 skin =
          a_weights_0.x * u_boneMatrices[a_joints_0.x]
        + a_weights_0.y * u_boneMatrices[a_joints_0.y]
        + a_weights_0.z * u_boneMatrices[a_joints_0.z]
        + a_weights_0.w * u_boneMatrices[a_joints_0.w];
    gl_Position = u_lightSpaceMatrix * skin * vec4(a_position, 1.0);
}
