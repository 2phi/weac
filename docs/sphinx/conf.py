# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'WEAC'
copyright = '2024, 2phi GbR'
author = 'P.L. Rosendahl, P. Weissgraeber, F. Rheinschmidt, J. Schneider'
release = '2.5.0'
github_url = 'https://github.com/2phi/weac'



# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinxawesome_theme.highlighting',
]

pygments_style = 'perldoc'
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
html_theme = 'sphinxawesome_theme'
html_theme_options = {
    'logo_light': '_static/logo-light.png',
    'logo_dark': '_static/logo-dark.png',
    'awesome_external_links': True,
    'awesome_headerlinks': True,
    'show_scrolltop': True,
}
html_favicon = '_static/favicon.ico'
html_show_sphinx = False