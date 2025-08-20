"""
Sphinx configuration for WEAC documentation.

This configuration avoids deprecated extensions and ensures that
`version` and `release` are strings (not callables) as required by Sphinx.
"""

from __future__ import annotations

import os
import sys
from importlib.metadata import version as get_version

# Add the project's 'src' directory to the path
sys.path.insert(0, os.path.abspath("../../src"))


# -- Project information -----------------------------------------------------

project = "WEAC"
author = "2phi"

# Ensure these are strings. Do not shadow the imported function name.
release = get_version("weac")
version = ".".join(release.split(".")[:2])


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

# Do NOT include 'sphinxawesome_theme.highlighting' (deprecated and unnecessary)

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

html_theme = "sphinxawesome_theme"
html_static_path = ["_static"]
html_title = f"{project} {release}"

# GitHub Pages configuration
html_baseurl = "https://2phi.github.io/weac/"
html_use_index = True

# Theme options for sphinxawesome_theme
html_theme_options = {
    "show_breadcrumbs": True,
    "breadcrumbs_separator": " > ",
}


# -- Autodoc options ---------------------------------------------------------

autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_preserve_defaults = True
autodoc_mock_imports = []
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Suppress warnings
suppress_warnings = [
    "app.add_node",
    "app.add_directive",
    "app.add_role",
    "app.add_generic_role",
    "app.add_source_parser",
    "download.not_readable",
    "image.not_readable",
    "ref.ref",
    "ref.numref",
    "ref.keyword",
    "ref.option",
    "misc.highlighting_failure",
]
