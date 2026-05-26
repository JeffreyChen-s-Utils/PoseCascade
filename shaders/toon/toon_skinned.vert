#version 410 core
//
// pass: toon / forward (skinned)
// attributes (location):
//   0 a_position    (vec3)
//   1 a_normal      (vec3)
//   3 a_texcoord_0  (vec2)
//   4 a_joints_0    (uvec4)
//   5 a_weights_0   (vec4)
// uniforms:
//   u_viewMatrix, u_projMatrix
//   u_boneMatrices[MAX_BONES] — pre-multiplied (joint world × inverse bind),
//     uploaded by the renderer once per frame per skinned mesh.

#define MAX_BONES 384

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 3) in vec2 a_texcoord_0;
layout(location = 4) in uvec4 a_joints_0;
layout(location = 5) in vec4 a_weights_0;

uniform mat4 u_viewMatrix;
uniform mat4 u_projMatrix;
uniform mat4 u_boneMatrices[MAX_BONES];
// Optional floor clamp — same semantics as forward/skinned.vert.
uniform int u_groundEnabled;
uniform float u_groundY;
uniform float u_groundTolerance;

out vec3 v_normal_world;
out vec3 v_normal_view;
out vec2 v_uv;
out vec3 v_world_pos;

void main() {
    mat4 skin =
          a_weights_0.x * u_boneMatrices[a_joints_0.x]
        + a_weights_0.y * u_boneMatrices[a_joints_0.y]
        + a_weights_0.z * u_boneMatrices[a_joints_0.z]
        + a_weights_0.w * u_boneMatrices[a_joints_0.w];

    vec4 skinned_position = skin * vec4(a_position, 1.0);
    vec3 skinned_normal = mat3(skin) * a_normal;
    vec3 n_world = normalize(skinned_normal);

    if (u_groundEnabled != 0) {
        // 1 mm z-fight epsilon — see forward/skinned.vert for rationale.
        float soft_floor = u_groundY - u_groundTolerance + 0.001;
        if (skinned_position.y < soft_floor) {
            skinned_position.y = soft_floor;
        }
    }

    v_normal_world = n_world;
    v_normal_view = normalize(mat3(u_viewMatrix) * n_world);
    v_uv = a_texcoord_0;
    v_world_pos = skinned_position.xyz;
    gl_Position = u_projMatrix * u_viewMatrix * skinned_position;
}
