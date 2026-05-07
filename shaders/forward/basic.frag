#version 410 core
//
// pass: forward / basic
// uniforms:
//   u_baseColor (vec4) — flat tint multiplier (PBR baseColorFactor)
//   u_baseColorTex (sampler2D) — bound to GL_TEXTURE0; renderer always binds
//     either the mesh's albedo map or a 1x1 white fallback, so the sample is
//     always valid.
//
// Lighting: a single directional light + a hemispherical ambient term so
// back-lit surfaces never collapse to pure black.

in vec3 v_normal;
in vec2 v_uv;

uniform vec4 u_baseColor;
uniform sampler2D u_baseColorTex;

out vec4 frag_color;

const vec3 LIGHT_DIR = normalize(vec3(0.3, 0.7, 0.6));
const float AMBIENT = 0.45;

void main() {
    vec3 normal = normalize(v_normal);
    float lambert = max(dot(normal, LIGHT_DIR), 0.0);
    float light = AMBIENT + (1.0 - AMBIENT) * lambert;
    vec4 sampled = texture(u_baseColorTex, v_uv);
    vec3 base = u_baseColor.rgb * sampled.rgb;
    frag_color = vec4(base * light, u_baseColor.a * sampled.a);
}
