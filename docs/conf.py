# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import sphinx_rtd_theme

# sys.path.insert(0, os.path.abspath(".."))
# sys.path.append(os.path.join(os.path.dirname(__name__), ".."))

# -- Project information -----------------------------------------------------

project = "pyrosm"
copyright = "2020, Henrikki Tenkanen"
author = "Henrikki Tenkanen"

# The full version, including alpha/beta/rc tags
version = release = "0.4.3"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx.ext.intersphinx",

    # Ipython
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",

    # NBsphinx
    "nbsphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "alabaster"

html_theme = 'pydata_sphinx_theme'

html_context = {
    # Enable the "Edit in GitHub link within the header of each page.
    'display_github': True,
    # Set the following variables to generate the resulting github URL for each page.
    # Format Template: https://{{ github_host|default("github.com") }}/{{ github_user }}/{{ github_repo }}/blob/{{ github_version }}{{ conf_py_path }}{{ pagename }}{{ suffix }}
    'github_user': 'HTenkanen',
    'github_repo': 'pyrosm',
    'github_version': 'master/',
    'conf_py_path': 'docs/'
}


# Theme options for sphinx_rtd_theme:
# -----------------------------------

# html_theme_options = {
#     "canonical_url": "",
#     "analytics_id": "UA-XXXXXXX-1",  #  Provided by Google in your dashboard
#     "logo_only": False,
#     "display_version": True,
#     "prev_next_buttons_location": "bottom",
#     "style_external_links": False,
#     "vcs_pageview_mode": "",
#     "style_nav_header_background": "white",
#     # Toc options
#     "collapse_navigation": True,
#     "sticky_navigation": True,
#     "navigation_depth": 4,
#     "includehidden": True,
#     "titles_only": False
# }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = [
    'css/custom.css',
]

# ======================
# Do not skip class init
# ======================

def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)
