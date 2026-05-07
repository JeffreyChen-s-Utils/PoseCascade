#version 410 core
//
// pass: toon / forward (shared between toon.vert and toon_skinned.vert)
// uniforms:
//   u_baseColor (vec4)        — diffuse tint (PMX material.diffuse)
//   u_baseColorTex (sampler2D, unit 0) — albedo / diffuse map
//   u_sphereTex (sampler2D, unit 1)    — sphere-texture map; renderer binds
//     a 1x1 neutral fallback when the material has no sphere reference.
//   u_toonTex (sampler2D, unit 2)      — 1x256 toon ramp; samples at
//     (0.5, 1 - lambert) so the top of the ramp is full light, the
//     bottom is full shadow.
//   u_sphereMode (int)        — 0 disabled, 1 multiply, 2 add, 3 sub-tex
//   u_specular (vec3)         — Blinn-Phong specular tint
//   u_specularPower (float)   — Blinn-Phong shininess exponent
//   u_ambient (vec3)          — flat ambient term added to the lit colour
//
// MMD's shading is not PBR — it's a Lambert × ramp + Blinn-Phong specular
// composite. Sphere textures composite multiplicatively or additively
// against the diffuse term using a view-space normal as the lookup UV
// (matches the spherical-environment "sph" / "spa" convention).

in vec3 v_normal_world;
in vec3 v_normal_view;
in vec2 v_uv;

uniform vec4 u_baseColor;
uniform sampler2D u_baseColorTex;
uniform sampler2D u_sphereTex;
uniform sampler2D u_toonTex;
uniform int u_sphereMode;
uniform vec3 u_specular;
uniform float u_specularPower;
uniform vec3 u_ambient;
uniform vec3 u_lightDirection;     // unit vector pointing toward the light source
uniform vec3 u_lightColor;         // RGB scale on the directional contribution

out vec4 frag_color;

const int SPHERE_DISABLED = 0;
const int SPHERE_MULTIPLY = 1;
const int SPHERE_ADD = 2;
const int SPHERE_SUB_TEXTURE = 3;

void main() {
    vec3 normal = normalize(v_normal_world);
    vec3 light_dir = normalize(u_lightDirection);
    float lambert = clamp(dot(normal, light_dir), 0.0, 1.0);

    // Toon ramp: top of the ramp = bright (lambert ≈ 1), bottom = dark
    // (lambert ≈ 0). Sampling at u=0.5 hits the column centre of the
    // 1×N ramp regardless of width.
    vec3 ramp = texture(u_toonTex, vec2(0.5, 1.0 - lambert)).rgb;

    vec4 albedo = texture(u_baseColorTex, v_uv);
    vec3 diffuse_term = u_baseColor.rgb * albedo.rgb * ramp * u_lightColor;

    // Sphere texture composite. The lookup is the view-space normal mapped
    // to [0,1]² which gives a stable spherical reflection regardless of
    // model orientation under camera rotation.
    vec2 sphere_uv = v_normal_view.xy * 0.5 + 0.5;
    vec3 sphere_sample = texture(u_sphereTex, sphere_uv).rgb;
    if (u_sphereMode == SPHERE_MULTIPLY) {
        diffuse_term *= sphere_sample;
    } else if (u_sphereMode == SPHERE_ADD) {
        diffuse_term += sphere_sample;
    } else if (u_sphereMode == SPHERE_SUB_TEXTURE) {
        // Sub-texture mode treats the second-UV channel as an overlay; we
        // do not yet ship a UV2 path, so we fall back to multiply for now
        // (visual approximation, kept neutral on toon ramp materials).
        diffuse_term *= sphere_sample;
    }

    // Blinn-Phong specular. Half vector against the view direction; we
    // approximate the view direction as the camera's forward axis in view
    // space, i.e. (0, 0, 1) — same convention MMD uses for its "default"
    // eye position.
    vec3 view_dir = vec3(0.0, 0.0, 1.0);
    vec3 half_dir = normalize(light_dir + view_dir);
    float spec = 0.0;
    if (u_specularPower > 0.0) {
        spec = pow(max(dot(normal, half_dir), 0.0), max(u_specularPower, 1.0));
    }
    vec3 specular_term = u_specular * spec;

    vec3 colour = diffuse_term + u_ambient + specular_term;
    frag_color = vec4(colour, u_baseColor.a * albedo.a);
}
