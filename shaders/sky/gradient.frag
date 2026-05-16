#version 410 core
//
// pass: sky / gradient
// uniforms:
//   u_zenithColor  (vec3) — colour at the top of the frame
//   u_horizonColor (vec3) — colour at the horizon line
//   u_groundColor  (vec3) — colour below the horizon
//   u_horizonY     (float) — screen-space Y of the horizon (default 0.5)
//
// Vertical three-stop gradient. ``u_horizonY`` lets the integrator
// shift the horizon up/down if the camera is high/low. The fragment
// produces sRGB-encoded values; combined with the renderer's
// ``GL_FRAMEBUFFER_SRGB`` it lands on screen exactly as specified.

in vec2 v_uv;

uniform vec3 u_zenithColor;
uniform vec3 u_horizonColor;
uniform vec3 u_groundColor;
uniform float u_horizonY;

out vec4 frag_color;

void main() {
    float t;
    vec3 colour;
    if (v_uv.y >= u_horizonY) {
        // Above horizon — fade zenith → horizon.
        t = clamp((v_uv.y - u_horizonY) / max(1.0 - u_horizonY, 1.0e-4), 0.0, 1.0);
        colour = mix(u_horizonColor, u_zenithColor, t);
    } else {
        // Below horizon — fade horizon → ground.
        t = clamp(1.0 - v_uv.y / max(u_horizonY, 1.0e-4), 0.0, 1.0);
        colour = mix(u_horizonColor, u_groundColor, t);
    }
    frag_color = vec4(colour, 1.0);
}
