#version 410 core
//
// pass: forward / skinned (dual-quaternion skinning)
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
// Default skinned shader for plain glTF / forward-rendered meshes (the
// path the editor's viewer takes for any skinned mesh that isn't routed
// through the toon pipeline). Switched from LBS to DQS so non-toon
// meshes also benefit from joint-volume preservation — the toon path
// already did this via toon_skinned_dqs.vert. The renderer's
// :meth:`_upload_skin_uniforms` falls through to the DQ uniforms when
// this program reports no ``u_boneMatrices`` location, so no host
// change is needed.

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
// Optional floor clamp — when ``u_groundEnabled != 0`` any skinned
// vertex whose world Y falls below ``u_groundY - u_groundTolerance``
// is lifted up to that soft floor. Lets the forward skinned path
// (used for body + clothes when cloth_host isn't owning them) stop
// the mesh clipping through the floor without round-tripping through
// cloth_host. Defaults to disabled (zero == off) so existing meshes
// that shouldn't be floor-clamped are unaffected.
uniform int u_groundEnabled;
uniform float u_groundY;
uniform float u_groundTolerance;

out vec3 v_normal;
out vec2 v_uv;

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
    vec3 skinned_position = rotated_p + t;
    vec3 skinned_normal = _rotate_by_quat(a_normal, blend_r);

    // Optional floor clamp. With cloth_host disabled for this mesh, no
    // CPU/GPU cloth pass clamps below-floor verts — without this branch
    // the skirt / coat trails through the ground in horizontal poses.
    if (u_groundEnabled != 0) {
        // Lift to soft floor PLUS a 1 mm z-fight epsilon — clamping to
        // exactly the ground plane leaves the body coplanar with the
        // ground mesh, so per-pixel depth ties flicker as small dark
        // dots poking through from below. The 1 mm offset is invisible
        // visually but resolves the depth tie cleanly.
        float soft_floor = u_groundY - u_groundTolerance + 0.001;
        if (skinned_position.y < soft_floor) {
            skinned_position.y = soft_floor;
        }
    }

    v_normal = normalize(skinned_normal);
    v_uv = a_texcoord_0;
    gl_Position = u_projMatrix * u_viewMatrix * vec4(skinned_position, 1.0);
}
