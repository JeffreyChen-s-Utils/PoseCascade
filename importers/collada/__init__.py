"""COLLADA (.dae) importer plugin.

COLLADA is XML, so the implementation MUST use ``defusedxml.ElementTree``
(not stdlib ``xml.etree``) to defend against XXE / billion-laughs attacks.
"""

from collada.importer import ColladaImporter

importer_class = ColladaImporter
