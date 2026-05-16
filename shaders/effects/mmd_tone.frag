#version 410 core
//
// pass: effect / mmd_tone (gamma-curve approximation)
// inputs:  u_main_color — previous pass's output texture
// uniforms:
//   midtone_lift  (float, default 0.04) — adds a small floor so the
//     darkest pixels never crush to pure black, matching MMD's
//     tendency to keep shadowed areas slightly grey.
//   highlight_rolloff (float, default 0.15) — soft-knee compression
//     applied above ~80% luma so direct-lit pixels don't clip the way
//     a naive sRGB output would.
//   saturation (float, default 1.08) — gentle saturation boost; MMD
//     tends to render slightly warmer / more colourful than a
//     calibrated sRGB pipeline does.
//   warm_tint (vec3, default (1.02, 1.0, 0.97)) — warm bias on the
//     final RGB, opt-in for the "MMD feel" cooling-grey calibrated
//     monitors lose.
//
// The curve is a single fragment-shader pass — no LUT texture needed.
// Results land back in the same colour space the input was in; if the
// engine's framebuffer is sRGB-aware the GPU re-encodes on write, so
// adding this effect on top of the default sRGB output is safe.

in vec2 v_uv;

uniform sampler2D u_main_color;
uniform float midtone_lift;
uniform float highlight_rolloff;
uniform float saturation;
uniform vec3 warm_tint;

out vec4 frag_color;

const vec3 LUMA_WEIGHTS = vec3(0.2126, 0.7152, 0.0722);

vec3 _apply_curve(vec3 c) {
    // Lift floor: ``c + midtone_lift * (1 - c)`` keeps the curve passing
    // through (1, 1) so highlights aren't shifted.
    vec3 lifted = c + midtone_lift * (1.0 - c);
    // Highlight rolloff: a smooth Reinhard-style knee above ``knee`` so
    // direct-lit pixels (bloom + key light) don't slam against the 1.0
    // ceiling.
    float knee = 1.0 - highlight_rolloff;
    vec3 hi = max(lifted - knee, 0.0);
    vec3 rolled = lifted - hi + hi / (1.0 + hi);
    return rolled;
}

void main() {
    vec4 base = texture(u_main_color, v_uv);
    vec3 c = _apply_curve(base.rgb);
    // Saturation: lerp from luminance-grey to the curve-adjusted colour.
    float luma = dot(c, LUMA_WEIGHTS);
    vec3 saturated = mix(vec3(luma), c, saturation);
    // Warm tint: multiplicative bias, keeps blacks black.
    vec3 tinted = saturated * warm_tint;
    frag_color = vec4(tinted, base.a);
}
