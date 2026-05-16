#version 410 core
//
// pass: forward / skinned
// attributes (location):
//   0 a_position    (vec3)
//   1 a_normal      (vec3)
//   3 a_texcoord_0  (vec2)
//   4 a_joints_0    (uvec4) — joint indices, blended by weights
//   5 a_weights_0   (vec4)
// uniforms:
//   u_viewMatrix, u_projMatrix
//   u_boneMatrices[MAX_BONES] — pre-multiplied (joint world × inverse bind),
//     uploaded by the renderer once per frame per skinned mesh.
//
// Bone matrices encode the skin's deformation directly in world space, so
// u_modelMatrix is intentionally absent here — the renderer pre-folds any
// holder transform into the bone matrices when it gathers them.

#define MAX_BONES 384

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 3) in vec2 a_texcoord_0;
layout(location = 4) in uvec4 a_joints_0;
layout(location = 5) in vec4 a_weights_0;

uniform mat4 u_viewMatrix;
uniform mat4 u_projMatrix;
uniform mat4 u_boneMatrices[MAX_BONES];

out vec3 v_normal;
out vec2 v_uv;

void main() {
    mat4 skin =
          a_weights_0.x * u_boneMatrices[a_joints_0.x]
        + a_weights_0.y * u_boneMatrices[a_joints_0.y]
        + a_weights_0.z * u_boneMatrices[a_joints_0.z]
        + a_weights_0.w * u_boneMatrices[a_joints_0.w];

    vec4 skinned_position = skin * vec4(a_position, 1.0);
    vec3 skinned_normal = mat3(skin) * a_normal;

    v_normal = normalize(skinned_normal);
    v_uv = a_texcoord_0;
    gl_Position = u_projMatrix * u_viewMatrix * skinned_position;
}
