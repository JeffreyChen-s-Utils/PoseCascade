#version 410 core
//
// pass: ground / projected shadow
// uniforms:
//   u_shadowColor    (vec4) — RGBA of the projected silhouette; alpha < 1
//                             so the checker shows through.
//   u_shadowMaxHeight (float) — vertices farther than this above the
//                               ground fade to zero alpha. Keeps the
//                               shadow a "contact" effect — a horizontal
//                               body in dog crawl only casts a shadow
//                               where it actually touches the floor,
//                               instead of an unbroken full-body streak.
//
// The projection lives in the vertex shader; this pass just fades.

uniform vec4 u_shadowColor;
uniform float u_shadowMaxHeight;

in float v_height_above_ground;

out vec4 frag_color;

void main() {
    // Hard cutoff: any fragment whose source vertex was farther above
    // the ground than ``u_shadowMaxHeight`` is dropped entirely. This
    // turns the projected shadow into a true CONTACT shadow — a
    // horizontal body (dog_crawl) only writes pixels under the bone
    // segments that are actually touching the floor (hands, knees,
    // feet), rather than a faded full-body streak.
    if (v_height_above_ground > u_shadowMaxHeight) {
        discard;
    }
    // Inside the contact band, fade smoothly from full alpha at the
    // floor to zero at the cutoff so the shadow's edge is soft rather
    // than a step that the user reads as "deformed".
    float fade = 1.0 - smoothstep(
        0.0, max(u_shadowMaxHeight, 1e-4), v_height_above_ground
    );
    frag_color = vec4(u_shadowColor.rgb, u_shadowColor.a * fade);
}
