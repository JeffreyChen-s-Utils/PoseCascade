#version 410 core
//
// pass: toon / outline (skinned inverted hull, dual-quaternion skinning)
// attributes (location):
//   0 a_position    (vec3)
//   1 a_normal      (vec3)
//   4 a_joints_0    (uvec4)
//   5 a_weights_0   (vec4)
// uniforms:
//   u_viewMatrix, u_projMatrix
//   u_boneDualQuatsReal[MAX_BONES] (vec4) — rotation quaternion (xyzw)
//   u_boneDualQuatsDual[MAX_BONES] (vec4) — dual quaternion (xyzw)
//   u_edgeSize (float)
//
// Skinned outline expansion. Switched from LBS to dual-quaternion
// skinning so the silhouette doesn't candy-wrap at twisted joints —
// previously the outline pass produced a visibly-pinched edge at the
// shoulder / waist / knee in horizontal poses even though the toon
// body fill (DQS) was smooth. The DQ uniforms are auto-uploaded by
// :meth:`Renderer._upload_skin_uniforms` because this program has no
// ``u_boneMatrices`` location, so the matrix path is skipped and the
// DQ pair is filled instead. Output: position + normal blended via
// DQS, then expanded along the normal by ``u_edgeSize``.

#define MAX_BONES 384

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 4) in uvec4 a_joints_0;
layout(location = 5) in vec4 a_weights_0;

uniform mat4 u_viewMatrix;
uniform mat4 u_projMatrix;
uniform vec4 u_boneDualQuatsReal[MAX_BONES];
uniform vec4 u_boneDualQuatsDual[MAX_BONES];
uniform float u_edgeSize;
// Optional floor clamp — same uniforms as forward/skinned.vert. Applied
// AFTER the outline expansion so the silhouette doesn't poke below the
// ground plane (otherwise the inflated foot sole / palm reaches under
// the floor and looks like a 1 cm halo sticking out).
uniform int u_groundEnabled;
uniform float u_groundY;
uniform float u_groundTolerance;

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

    // Antipodal correction — see toon_skinned_dqs.vert for the same
    // pattern. Flipping signs keeps each contributing bone on the
    // first bone's hemisphere so the weighted blend doesn't cancel
    // at twisted joints.
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
    vec3 skinned_position = rotated_p + t;
    vec3 skinned_normal = normalize(_rotate_by_quat(a_normal, blend_r));
    vec3 expanded_world = skinned_position + skinned_normal * u_edgeSize;
    // Floor clamp the EXPANDED silhouette so the outline doesn't poke
    // below the ground when the sole / palm normal points downward.
    if (u_groundEnabled != 0) {
        // 1 mm z-fight epsilon — see forward/skinned.vert for rationale.
        float soft_floor = u_groundY - u_groundTolerance + 0.001;
        if (expanded_world.y < soft_floor) {
            expanded_world.y = soft_floor;
        }
    }
    gl_Position = u_projMatrix * u_viewMatrix * vec4(expanded_world, 1.0);
}
