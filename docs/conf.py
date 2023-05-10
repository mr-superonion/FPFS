# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# flake8: noqa

import os
import sys

sys.path.insert(0, os.path.abspath("../fpfs"))

import fpfs

# -- Project information -----------------------------------------------------

project = "FPFS"
copyright = "2022, Xiangchong Li"
author = "Xiangchong Li"

# The full version, including alpha/beta/rc tags
release = fpfs.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_book_theme",
    "myst_nb",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org", None),
    "numpy": ("http://docs.scipy.org/doc/numpy", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("http://matplotlib.org/stable", None),
    "myst": ("https://myst-parser.readthedocs.io/en/latest/", None),
    "markdown_it": ("https://markdown-it-py.readthedocs.io/en/latest", None),
    "nbclient": ("https://nbclient.readthedocs.io/en/latest", None),
    "nbformat": ("https://nbformat.readthedocs.io/en/latest", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
}

autosectionlabel_prefix_document = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True
napoleon_custom_sections = [("Returns", "params_style")]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".myst": "myst-nb",
    ".ipynb": "myst-nb",
}
master_doc = "index"
nb_merge_streams = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'sphinx_rtd_theme'
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme = "sphinx_book_theme"
html_theme_options = {
    "icon_links": [],
    "github_url": "https://github.com/executablebooks/myst-nb",
    "repository_url": "https://github.com/executablebooks/myst-nb",
    "repository_branch": "master",
    "home_page_in_toc": True,
    "path_to_docs": "docs",
    "show_navbar_depth": 1,
    "use_edit_page_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "classic",
    },
}

# def setup(app):
#     app.add_css_file('xtheme.css')

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "any"

# myst
myst_enable_extensions = [
    "colon_fence",
    "amsmath",
    "dollarmath",
    "deflist",
    "fieldlist",
    "tasklist",
    "smartquotes",
    "html_admonition",
    "html_image",
]
myst_number_code_blocks = ["typescript"]
myst_heading_anchors = 2
myst_footnote_transition = True
myst_dmath_double_inline = True

htmlhelp_basename = "fpfs_doc"
