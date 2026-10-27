# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
import os
from datetime import datetime
from pathlib import Path

current_year = datetime.now().year

project = "pyrosm"
copyright = f"2020-{current_year}, Henrikki Tenkanen + pyrosm contributors"
author = "Henrikki Tenkanen + pyrosm contributors"

# The full version, including alpha/beta/rc tags
version = release = "0.26.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    # Generate the API reference as category tables with one-line summaries and a page per
    # method/function (see reference.rst).
    "sphinx.ext.autosummary",
    # Support numpy style autodoc
    "sphinx.ext.napoleon",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "myst_nb",
    "sphinx_design",
    # Write llms.txt, an index of the pages for language models.
    "sphinx_llms_txt",
]

# Generate the per-object stub pages referenced by the autosummary tables at build time.
autosummary_generate = True

# Enable MyST's colon-fence syntax (:::{admonition} ... :::) so notebook markdown
# cells can use admonitions/directives that also render cleanly in the notebook UI.
myst_enable_extensions = ["colon_fence"]

# Generate GitHub-style anchors for headings up to level 3 so the in-page "Contents"
# links inside the notebooks (e.g. ``[...](#read-buildings)``) resolve to their sections.
myst_heading_anchors = 3

# pyrosm is installed (and compiled) by the Read the Docs build via
# ``.readthedocs.yml`` (``python.install`` runs ``pip install .``), so autodoc
# imports the real package and reads docstrings straight from it -- no mocking
# of the compiled Cython extensions or binary deps is needed.


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

html_theme = "sphinx_book_theme"
html_title = ""
html_logo = "img/pyrosm_logo.png"

html_theme_options = {
    # "external_links": [],
    "repository_url": "https://github.com/HTenkanen/pyrosm/",
    "repository_branch": "main",
    "path_to_docs": "docs/",
    # "twitter_url": "https://twitter.com/pythongis",
    # "google_analytics_id": "UA-159257488-1",
    "use_edit_page_button": True,
    "use_repository_button": True,
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "jupyterlab",
        "collapse_navigation": False,
    },
}

html_context = {
    # Enable the "Edit in GitHub link within the header of each page.
    "display_github": True,
    # Set the following variables to generate the resulting github URL for each page.
    # Format Template: https://{{ github_host|default("github.com") }}/{{ github_user }}/{{ github_repo }}/blob/{{ github_version }}{{ conf_py_path }}{{ pagename }}{{ suffix }}
    "github_user": "htenkanen",
    "github_repo": "pyrosm",
    "github_version": "main/",
    "conf_py_path": "/docs/",
}

# The master toctree document.
master_doc = "index"

# The site's address, so that llms.txt links to the published pages (Read the Docs sets it).
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
llms_txt_summary = (
    "pyrosm reads OpenStreetMap PBF files into GeoPandas GeoDataFrames and downloads "
    "OpenStreetMap extracts from Geofabrik, BBBike and Movisda. Before writing code that "
    "downloads data or geocodes places, read the Download responsibly page "
    "(%sdownload_responsibly.html) and follow its rules for AI coding agents." % html_baseurl
)
# Link the rendered pages, which hold the API docstrings and the notebooks without their
# images' data. No llms-full.txt, the sources in one file, which would be mostly notebook
# JSON: sphinx-llms-txt 0.7 writes it even with llms_txt_full_file = False, and skips it
# only over its size limit.
llms_txt_uri_template = ("{base_url}" if html_baseurl else "") + "{docname}.html"
llms_txt_full_file = False
llms_txt_full_max_size = 0
llms_txt_full_size_policy = "info_skip"


def _policy_page_first(app, exception):
    """Move the Download responsibly page to the top of the list of pages in llms.txt."""
    path = Path(app.outdir) / "llms.txt"
    if exception is not None or not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    policy = [line for line in lines if line.startswith("- [Download responsibly](")]
    rest = [line for line in lines if line not in policy]
    first = next(i for i, line in enumerate(rest) if line.startswith("- ["))
    path.write_text("".join(rest[:first] + policy + rest[first:]), encoding="utf-8")


def setup(app):
    # After sphinx-llms-txt, which writes llms.txt at the default priority, 500.
    app.connect("build-finished", _policy_page_first, priority=900)

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# Render notebooks from their stored outputs; never execute them at build time.
nb_execution_mode = "off"
nb_execution_allow_errors = True
