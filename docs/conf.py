# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# sys.path.insert(0, os.path.abspath(".."))
# sys.path.append(os.path.join(os.path.dirname(__name__), ".."))

# -- Project information -----------------------------------------------------

project = "pyrosm"
copyright = "2020, Henrikki Tenkanen"
author = "Henrikki Tenkanen"

# The full version, including alpha/beta/rc tags
version = release = "0.6.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.

extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    "sphinx_thebe",
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'myst_nb',
    'jupyter_sphinx',
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

html_theme = 'sphinx_book_theme'
html_title = ""
html_logo = 'img/logo.PNG'

html_theme_options = {
    #"external_links": [],
    "repository_url": "https://github.com/HTenkanen/pyrosm/",
    "repository_branch": "master",
    "path_to_docs": "docs/",
    #"twitter_url": "https://twitter.com/pythongis",
    #"google_analytics_id": "UA-159257488-1",
    "use_edit_page_button": True,
    "use_repository_button": True,
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "thebe": False,
        "notebook_interface": "jupyterlab",
    "collapse_navigation" : False
    },
}

html_context = {
    # Enable the "Edit in GitHub link within the header of each page.
    'display_github': True,
    # Set the following variables to generate the resulting github URL for each page.
    # Format Template: https://{{ github_host|default("github.com") }}/{{ github_user }}/{{ github_repo }}/blob/{{ github_version }}{{ conf_py_path }}{{ pagename }}{{ suffix }}
    'github_user': 'htenkanen',
    'github_repo': 'pyrosm',
    'github_version': 'master/',
    'conf_py_path': '/docs/'
}

# The master toctree document.
master_doc = 'index'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

#html_css_files = ['css/custom.css']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# Allow errors
execution_allow_errors = True

# Do not execute cells
jupyter_execute_notebooks = "off"

# Allow myst admonition style
myst_admonition_enable = True
