"""
Sphinx configuration for WEAC documentation.

This configuration avoids deprecated extensions and ensures that
`version` and `release` are strings (not callables) as required by Sphinx.
"""

from __future__ import annotations

from importlib.metadata import version as get_version


# -- Project information -----------------------------------------------------

project = "WEAC"
author = "2phi GbR"

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


# -- Autodoc options ---------------------------------------------------------

autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_preserve_defaults = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
