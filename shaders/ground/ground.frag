#version 410 core
//
// pass: ground / checker
// uniforms:
//   u_cellSize  (float) — world-space size of one checker cell, default 0.5 m
//   u_colorA    (vec4)  — first checker shade (light)
//   u_colorB    (vec4)  — second checker shade (dark)
//   u_fadeStart (float) — radius where the ground starts fading to ``u_colorA``
//                          alpha=0, so the floor blends into the background
//                          near the horizon instead of cutting off hard.
//   u_fadeEnd   (float) — radius where the fade reaches full transparency
//
// The classic MMD opening shows the model standing on a flat checkered
// floor — we reproduce it procedurally so the engine doesn't need a
// stage asset to look "right" on first launch.

in vec2 v_world_xz;

uniform float u_cellSize;
uniform vec4 u_colorA;
uniform vec4 u_colorB;
uniform float u_fadeStart;
uniform float u_fadeEnd;

out vec4 frag_color;

void main() {
    vec2 cell = floor(v_world_xz / u_cellSize);
    float parity = mod(cell.x + cell.y, 2.0);
    vec4 base = mix(u_colorA, u_colorB, parity);
    float distance_from_origin = length(v_world_xz);
    float fade = clamp(
        1.0 - (distance_from_origin - u_fadeStart) / max(u_fadeEnd - u_fadeStart, 1.0e-4),
        0.0, 1.0
    );
    frag_color = vec4(base.rgb, base.a * fade);
}
