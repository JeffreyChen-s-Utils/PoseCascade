"""DirectX .x (text) importer plugin.

Covers the static-stage subset of the format that the MMD ecosystem
actually emits — Frame hierarchy, FrameTransformMatrix, Mesh,
MeshNormals, MeshTextureCoords, MeshMaterialList. Skinning + animation
templates are ignored (warm path for stage / accessory imports).
"""

from x.importer import XImporter

importer_class = XImporter
