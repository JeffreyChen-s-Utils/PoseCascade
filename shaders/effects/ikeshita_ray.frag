#version 410 core
//
// pass: effect / ikeshita_ray (god rays)
// inputs: u_main_color (sampler2D)
// uniforms:
//   light_screen_position (vec3) — xy in screen NDC, z ignored
//   decay                 (float) — per-step radial decay
//   weight                (float) — per-sample contribution
//   exposure              (float) — final additive scale
//
// Standard radial blur from a light origin. Sample N steps along the
// vector from the current pixel toward the light origin, decaying
// each tap. The result composites additively over the scene.

in vec2 v_uv;

uniform sampler2D u_main_color;
uniform vec3 light_screen_position;
uniform float decay;
uniform float weight;
uniform float exposure;

out vec4 frag_color;

const int SAMPLES = 64;
const float DENSITY = 0.95;

void main() {
    vec2 light_uv = light_screen_position.xy;
    vec2 delta = (v_uv - light_uv) * (DENSITY / float(SAMPLES));
    vec2 sample_uv = v_uv;
    float intensity_scale = 1.0;
    vec3 accum = vec3(0.0);
    for (int i = 0; i < SAMPLES; i++) {
        sample_uv -= delta;
        vec3 sampled = texture(u_main_color, sample_uv).rgb * intensity_scale;
        accum += sampled * weight;
        intensity_scale *= decay;
    }
    vec3 base = texture(u_main_color, v_uv).rgb;
    frag_color = vec4(base + accum * exposure / float(SAMPLES), 1.0);
}
