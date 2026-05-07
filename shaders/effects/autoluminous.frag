#version 410 core
//
// pass: effect / autoluminous (emissive bloom)
// inputs: u_main_color (sampler2D) — bound to the previous pass's
//   output texture by the chain executor.
// uniforms:
//   threshold  (float)  — luminance gate
//   intensity  (float)  — final scale
//   tint       (vec3)   — bloom tint
//
// Minimal single-pass approximation: extract bright pixels by a luma
// threshold + add them back over the original. A higher-quality
// variant blurs the bright pass between extract and composite — the
// chain executor can chain a separate "blur" pass when one ships.

in vec2 v_uv;

uniform sampler2D u_main_color;
uniform float threshold;
uniform float intensity;
uniform vec3 tint;

out vec4 frag_color;

const vec3 LUMA_WEIGHTS = vec3(0.2126, 0.7152, 0.0722);

void main() {
    vec4 base = texture(u_main_color, v_uv);
    float luminance = dot(base.rgb, LUMA_WEIGHTS);
    float over = max(luminance - threshold, 0.0);
    vec3 bloom = base.rgb * over * intensity * tint;
    frag_color = vec4(base.rgb + bloom, base.a);
}
