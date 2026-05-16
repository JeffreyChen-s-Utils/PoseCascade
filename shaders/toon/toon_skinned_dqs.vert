#version 410 core
//
// pass: toon / forward (skinned, dual-quaternion skinning)
// attributes (location):
//   0 a_position    (vec3)
//   1 a_normal      (vec3)
//   3 a_texcoord_0  (vec2)
//   4 a_joints_0    (uvec4)
//   5 a_weights_0   (vec4)
// uniforms:
//   u_viewMatrix, u_projMatrix
//   u_boneDualQuatsReal[MAX_BONES] (vec4) — rotation quaternion (xyzw)
//   u_boneDualQuatsDual[MAX_BONES] (vec4) — dual quaternion (xyzw)
//
// Dual-quaternion skinning avoids the LBS candy-wrapper artefact at
// twisted joints by interpolating bone transforms as screw motions.
// Each blended DQ is renormalised so the result represents a unit
// rigid motion regardless of the weights' numerical noise.

#define MAX_BONES 384

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 3) in vec2 a_texcoord_0;
layout(location = 4) in uvec4 a_joints_0;
layout(location = 5) in vec4 a_weights_0;

uniform mat4 u_viewMatrix;
uniform mat4 u_projMatrix;
uniform vec4 u_boneDualQuatsReal[MAX_BONES];
uniform vec4 u_boneDualQuatsDual[MAX_BONES];

out vec3 v_normal_world;
out vec3 v_normal_view;
out vec2 v_uv;
out vec3 v_world_pos;

vec3 _rotate_by_quat(vec3 v, vec4 q) {
    // p' = p + 2 * cross(q.xyz, cross(q.xyz, p) + q.w * p)
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

void main() {
    vec4 q0r = u_boneDualQuatsReal[a_joints_0.x];
    vec4 q0d = u_boneDualQuatsDual[a_joints_0.x];
    vec4 q1r = u_boneDualQuatsReal[a_joints_0.y];
    vec4 q1d = u_boneDualQuatsDual[a_joints_0.y];
    vec4 q2r = u_boneDualQuatsReal[a_joints_0.z];
    vec4 q2d = u_boneDualQuatsDual[a_joints_0.z];
    vec4 q3r = u_boneDualQuatsReal[a_joints_0.w];
    vec4 q3d = u_boneDualQuatsDual[a_joints_0.w];

    // Antipodal correction: the dot product of antipodal quaternions is
    // negative — flipping signs keeps every contributing bone on the
    // same hemisphere as the first, so the weighted blend doesn't cancel
    // to zero at twisted joints.
    if (dot(q0r, q1r) < 0.0) { q1r = -q1r; q1d = -q1d; }
    if (dot(q0r, q2r) < 0.0) { q2r = -q2r; q2d = -q2d; }
    if (dot(q0r, q3r) < 0.0) { q3r = -q3r; q3d = -q3d; }

    vec4 blend_r = a_weights_0.x * q0r + a_weights_0.y * q1r
                 + a_weights_0.z * q2r + a_weights_0.w * q3r;
    vec4 blend_d = a_weights_0.x * q0d + a_weights_0.y * q1d
                 + a_weights_0.z * q2d + a_weights_0.w * q3d;

    float norm = max(length(blend_r), 1.0e-6);
    blend_r /= norm;
    blend_d /= norm;

    vec3 rotated_p = _rotate_by_quat(a_position, blend_r);
    vec3 t = 2.0 * (
        blend_r.w * blend_d.xyz - blend_d.w * blend_r.xyz
        + cross(blend_r.xyz, blend_d.xyz)
    );
    vec3 world_pos = rotated_p + t;

    vec3 rotated_n = _rotate_by_quat(a_normal, blend_r);
    vec3 n_world = normalize(rotated_n);

    v_normal_world = n_world;
    v_normal_view = normalize(mat3(u_viewMatrix) * n_world);
    v_uv = a_texcoord_0;
    v_world_pos = world_pos;
    gl_Position = u_projMatrix * u_viewMatrix * vec4(world_pos, 1.0);
}
