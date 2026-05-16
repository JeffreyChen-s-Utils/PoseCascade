#version 410 core
//
// pass: toon / outline (skinned inverted hull)

#define MAX_BONES 384

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 4) in uvec4 a_joints_0;
layout(location = 5) in vec4 a_weights_0;

uniform mat4 u_viewMatrix;
uniform mat4 u_projMatrix;
uniform mat4 u_boneMatrices[MAX_BONES];
uniform float u_edgeSize;

void main() {
    mat4 skin =
          a_weights_0.x * u_boneMatrices[a_joints_0.x]
        + a_weights_0.y * u_boneMatrices[a_joints_0.y]
        + a_weights_0.z * u_boneMatrices[a_joints_0.z]
        + a_weights_0.w * u_boneMatrices[a_joints_0.w];

    vec4 skinned_position = skin * vec4(a_position, 1.0);
    vec3 skinned_normal = normalize(mat3(skin) * a_normal);
    vec3 expanded_world = skinned_position.xyz + skinned_normal * u_edgeSize;
    gl_Position = u_projMatrix * u_viewMatrix * vec4(expanded_world, 1.0);
}
