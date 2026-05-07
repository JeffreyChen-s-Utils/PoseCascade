"""String-constant catalogue for shader uniform and attribute names.

Every shader uniform name lives here so duplicate string literals never appear
across passes (SonarQube ``python:S1192``).
"""
from __future__ import annotations

# Per-frame
U_VIEW_MATRIX = "u_viewMatrix"
U_PROJECTION_MATRIX = "u_projMatrix"
U_VIEW_POSITION = "u_viewPos"
U_TIME = "u_time"

# Per-object
U_MODEL_MATRIX = "u_modelMatrix"
U_NORMAL_MATRIX = "u_normalMatrix"
U_BONE_MATRICES = "u_boneMatrices"

# Material
U_BASE_COLOR = "u_baseColor"
U_METALLIC = "u_metallic"
U_ROUGHNESS = "u_roughness"
U_BASE_COLOR_TEX = "u_baseColorTex"
U_NORMAL_TEX = "u_normalTex"
U_METALLIC_ROUGHNESS_TEX = "u_metallicRoughnessTex"

# MMD material (toon pass)
U_SPHERE_TEX = "u_sphereTex"
U_TOON_TEX = "u_toonTex"
U_SPHERE_MODE = "u_sphereMode"
U_SPECULAR = "u_specular"
U_SPECULAR_POWER = "u_specularPower"
U_AMBIENT = "u_ambient"
U_EDGE_COLOR = "u_edgeColor"
U_EDGE_SIZE = "u_edgeSize"
U_LIGHT_DIRECTION = "u_lightDirection"
U_LIGHT_COLOR = "u_lightColor"

# Vertex attribute locations
A_POSITION = 0
A_NORMAL = 1
A_TANGENT = 2
A_TEXCOORD_0 = 3
A_JOINTS_0 = 4
A_WEIGHTS_0 = 5
