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
U_BONE_DQ_REAL = "u_boneDualQuatsReal"
U_BONE_DQ_DUAL = "u_boneDualQuatsDual"

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
U_SECONDARY_LIGHT_COUNT = "u_secondaryLightCount"
U_SECONDARY_LIGHT_DIRECTIONS = "u_secondaryLightDirections"
U_SECONDARY_LIGHT_COLORS = "u_secondaryLightColors"

# Ground pass (checker)
U_CELL_SIZE = "u_cellSize"
U_COLOR_A = "u_colorA"
U_COLOR_B = "u_colorB"
U_FADE_START = "u_fadeStart"
U_FADE_END = "u_fadeEnd"
U_SHADOW_COLOR = "u_shadowColor"
U_GROUND_Y = "u_groundY"

# Self-shadow map (depth pass + toon sampling)
U_LIGHT_SPACE_MATRIX = "u_lightSpaceMatrix"
U_SHADOW_MAP = "u_shadowMap"
U_SHADOW_ENABLED = "u_shadowEnabled"
U_SHADOW_STRENGTH = "u_shadowStrength"

# Gradient sky pass
U_ZENITH_COLOR = "u_zenithColor"
U_HORIZON_COLOR = "u_horizonColor"
U_GROUND_COLOR = "u_groundColor"
U_HORIZON_Y = "u_horizonY"

# Vertex attribute locations
A_POSITION = 0
A_NORMAL = 1
A_TANGENT = 2
A_TEXCOORD_0 = 3
A_JOINTS_0 = 4
A_WEIGHTS_0 = 5
