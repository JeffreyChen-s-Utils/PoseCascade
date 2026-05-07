"""FBX importer plugin (heavy optional dependency).

Requires the FBX SDK or a Python binding such as ``pyfbx``. Failing imports
mean the plugin is silently skipped at discovery time, which is the correct
plugin pattern (see CLAUDE.md → Engine Core vs Importers).
"""

from fbx.importer import FbxImporter

importer_class = FbxImporter
