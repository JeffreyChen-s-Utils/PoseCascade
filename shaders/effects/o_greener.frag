#version 410 core
//
// pass: effect / o_greener (colour grading)
// inputs: u_main_color (sampler2D)
// uniforms:
//   hue_shift_degrees (float)
//   saturation        (float)
//   contrast          (float)
//   lift              (vec3)
//
// Hue rotation is implemented in HSL via a matrix; saturation /
// contrast are textbook scalar manipulations against a luma reference.
// Standard cinematic grading workflow — same building blocks o_Greener
// uses, ported to GLSL.

in vec2 v_uv;

uniform sampler2D u_main_color;
uniform float hue_shift_degrees;
uniform float saturation;
uniform float contrast;
uniform vec3 lift;

out vec4 frag_color;

const vec3 LUMA_WEIGHTS = vec3(0.299, 0.587, 0.114);
const float DEG_TO_RAD = 0.017453292519943295;
const float MID_GREY = 0.5;

mat3 hue_rotation(float radians_) {
    float c = cos(radians_);
    float s = sin(radians_);
    // Approximation that keeps luma invariant — good enough for stylised
    // toon scenes without a full HSV round-trip.
    float k = 1.0 / 3.0;
    return mat3(
        c + (1.0 - c) * k,           (1.0 - c) * k - 0.5774 * s,  (1.0 - c) * k + 0.5774 * s,
        (1.0 - c) * k + 0.5774 * s,  c + (1.0 - c) * k,           (1.0 - c) * k - 0.5774 * s,
        (1.0 - c) * k - 0.5774 * s,  (1.0 - c) * k + 0.5774 * s,  c + (1.0 - c) * k
    );
}

void main() {
    vec3 colour = texture(u_main_color, v_uv).rgb;
    colour = hue_rotation(hue_shift_degrees * DEG_TO_RAD) * colour;
    float luma = dot(colour, LUMA_WEIGHTS);
    colour = mix(vec3(luma), colour, saturation);
    colour = (colour - MID_GREY) * contrast + MID_GREY;
    colour += lift;
    frag_color = vec4(colour, 1.0);
}
