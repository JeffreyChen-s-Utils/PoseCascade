#version 410 core
//
// pass: effect / hgshadow (improved shadow tint)
// inputs:
//   u_main_color (sampler2D) — scene colour
//   u_main_depth (sampler2D) — scene depth 0..1
// uniforms:
//   darken         (float)  — 0..1 mute factor
//   shadow_tint    (vec3)   — colour cast inside shadow
//   depth_falloff  (float)  — depth-edge sharpness
//
// Phase 13 ships a stub: a uniform tint over the whole scene scaled
// by a depth-derived shadow-mask. The full HgShadow algorithm reads
// the shadow map plus normal+light direction; that lands when the
// renderer grows a shadow-map pass.

in vec2 v_uv;

uniform sampler2D u_main_color;
uniform sampler2D u_main_depth;
uniform float darken;
uniform vec3 shadow_tint;
uniform float depth_falloff;

out vec4 frag_color;

void main() {
    vec4 base = texture(u_main_color, v_uv);
    float depth = texture(u_main_depth, v_uv).r;
    float shadow = clamp(1.0 - depth * depth_falloff, 0.0, 1.0);
    vec3 tinted = mix(base.rgb, base.rgb * shadow_tint, shadow * darken);
    frag_color = vec4(tinted, base.a);
}
