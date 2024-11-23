# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

module_path = os.path.abspath(os.path.join('..','..','sbm'))
sys.path.insert(0, module_path)

project = 'sbm'
copyright = '2024, Yusuke Takase'
author = "Yusuke Takase"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    #'nbsphinx',
    'sphinx.ext.autosectionlabel',
    'pydata_sphinx_theme',
]

autosummary_generate = True
autosummary_imported_members = True
autosectionlabel_prefix_document = True
autoclass_content = "class"
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
#html_theme = 'alabaster'
