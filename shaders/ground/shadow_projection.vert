#version 410 core
//
// pass: ground / projected shadow (non-skinned)
// attributes (location):
//   0 a_position (vec3)
// uniforms:
//   u_modelMatrix, u_viewMatrix, u_projMatrix
//   u_lightDirection (vec3) — unit vector pointing TOWARD the light source
//   u_groundY        (float) — Y coordinate of the ground plane (default 0)
//
// Cheap "stencil-on-floor" shadow: instead of doing a depth-buffer pass
// we project each vertex along the light direction onto the ground
// plane, then rasterise with the colour shader writing a uniform dark
// alpha. The result is the model's silhouette flattened against the
// floor — recognisable as MMD's classic ground shadow without needing
// a shadow map.

layout(location = 0) in vec3 a_position;

uniform mat4 u_modelMatrix;
uniform mat4 u_viewMatrix;
uniform mat4 u_projMatrix;
uniform vec3 u_lightDirection;
uniform float u_groundY;

out float v_height_above_ground;

const float _MIN_LIGHT_Y = 0.05;
const float _SHADOW_Y_OFFSET = 0.001;     // lifts the shadow a hair off the
                                          // ground so it wins the z-fight
                                          // with the checker plane.

void main() {
    vec4 world = u_modelMatrix * vec4(a_position, 1.0);
    // Project ``world`` along ``-u_lightDirection`` onto the y = u_groundY
    // plane. Clamp the light's Y component away from zero to avoid the
    // division blowing up when the light is nearly horizontal.
    float ly = max(u_lightDirection.y, _MIN_LIGHT_Y);
    float t = (world.y - u_groundY) / ly;
    vec3 projected = vec3(
        world.x - t * u_lightDirection.x,
        u_groundY + _SHADOW_Y_OFFSET,
        world.z - t * u_lightDirection.z
    );
    // Per-vertex height above the ground plane — the fragment shader
    // fades the silhouette alpha by this so a horizontal body (dog
    // crawl) only casts a strong shadow near its contact points
    // (hands / knees / feet) instead of an unbroken full-body streak.
    v_height_above_ground = max(world.y - u_groundY, 0.0);
    gl_Position = u_projMatrix * u_viewMatrix * vec4(projected, 1.0);
}
