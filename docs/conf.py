# -- Sphinx configuration for PoseCascade documentation --

project = "PoseCascade"
author = "PoseCascade Contributors"
copyright = "2025-2026, PoseCascade Contributors"
release = "0.1"

extensions = [
    "sphinx.ext.autosectionlabel",
    "myst_parser",
]

# Some section titles repeat across language indexes; namespace them by
# document path so RTD's link checker stops complaining about dupes.
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 3

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Markdown source files (rendering_pipeline.md, declarative_animation.md,
# mcp.md) ship side-by-side with the RST language indexes — myst_parser
# handles the .md → docutils conversion so they slot into the toctree.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "sphinx_rtd_theme"
html_static_path: list[str] = []
html_logo = None
html_favicon = None

# Don't choke on JSON code fences that use literal "..." placeholders —
# the rendering_pipeline.md and declarative_animation.md examples use
# ellipses inside JSON to mean "snipped". Treat them as plain text.
highlight_options = {"json": {"ensurenl": False}}

# -- Internationalisation ----------------------------------------------------
language = "en"
locale_dirs = ["locale/"]
gettext_compact = False

# -- Options for HTML output -------------------------------------------------
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "titles_only": False,
}
