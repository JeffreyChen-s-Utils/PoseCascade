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
//   u_groundEnabled / u_groundY / u_groundTolerance — optional floor
//     clamp matching forward/skinned.vert. Without it, body verts at
//     Y < ground end up in the shadow depth map; the toon pass then
//     samples those depths and marks the floor as self-shadowed where
//     the body "punched through" the ground plane — visible as small
//     dark dots scattered across the floor under the character.

#define MAX_BONES 384

layout(location = 0) in vec3 a_position;
layout(location = 4) in uvec4 a_joints_0;
layout(location = 5) in vec4 a_weights_0;

uniform mat4 u_lightSpaceMatrix;
uniform mat4 u_boneMatrices[MAX_BONES];
uniform int u_groundEnabled;
uniform float u_groundY;
uniform float u_groundTolerance;

void main() {
    mat4 skin =
          a_weights_0.x * u_boneMatrices[a_joints_0.x]
        + a_weights_0.y * u_boneMatrices[a_joints_0.y]
        + a_weights_0.z * u_boneMatrices[a_joints_0.z]
        + a_weights_0.w * u_boneMatrices[a_joints_0.w];
    vec4 skinned_position = skin * vec4(a_position, 1.0);
    if (u_groundEnabled != 0) {
        // 1 mm z-fight epsilon — see forward/skinned.vert for rationale.
        float soft_floor = u_groundY - u_groundTolerance + 0.001;
        if (skinned_position.y < soft_floor) {
            skinned_position.y = soft_floor;
        }
    }
    gl_Position = u_lightSpaceMatrix * skinned_position;
}
