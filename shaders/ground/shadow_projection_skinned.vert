#version 410 core
//
// pass: ground / projected shadow (skinned, dual-quaternion skinning)
// attributes (location):
//   0 a_position  (vec3)
//   4 a_joints_0  (uvec4)
//   5 a_weights_0 (vec4)
// uniforms:
//   u_viewMatrix, u_projMatrix
//   u_boneDualQuatsReal[MAX_BONES] (vec4) — rotation quaternion (xyzw)
//   u_boneDualQuatsDual[MAX_BONES] (vec4) — dual quaternion (xyzw)
//   u_lightDirection (vec3)
//   u_groundY (float)
//
// Switched from LBS to DQS so the projected silhouette no longer
// candy-wraps at twisted joints — the floor shadow used to show a
// pinched outline at the shoulder / waist / knee in horizontal
// poses even when the toon body fill (DQS) was smooth. The DQ
// uniforms are auto-uploaded by :meth:`Renderer._upload_skin_uniforms`
// because this program has no ``u_boneMatrices`` location.

#define MAX_BONES 384

layout(location = 0) in vec3 a_position;
layout(location = 4) in uvec4 a_joints_0;
layout(location = 5) in vec4 a_weights_0;

uniform mat4 u_viewMatrix;
uniform mat4 u_projMatrix;
uniform vec4 u_boneDualQuatsReal[MAX_BONES];
uniform vec4 u_boneDualQuatsDual[MAX_BONES];
uniform vec3 u_lightDirection;
uniform float u_groundY;

out float v_height_above_ground;

const float _MIN_LIGHT_Y = 0.05;
const float _SHADOW_Y_OFFSET = 0.001;

vec3 _rotate_by_quat(vec3 v, vec4 q) {
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
    vec3 world = rotated_p + t;

    float ly = max(u_lightDirection.y, _MIN_LIGHT_Y);
    float ts = (world.y - u_groundY) / ly;
    vec3 projected = vec3(
        world.x - ts * u_lightDirection.x,
        u_groundY + _SHADOW_Y_OFFSET,
        world.z - ts * u_lightDirection.z
    );
    v_height_above_ground = max(world.y - u_groundY, 0.0);
    gl_Position = u_projMatrix * u_viewMatrix * vec4(projected, 1.0);
}
