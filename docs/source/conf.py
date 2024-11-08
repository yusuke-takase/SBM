# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import toml

module_path = os.path.abspath(os.path.join('..','..','sbm'))
print("module_path: ", module_path)
sys.path.insert(0, module_path)
pyproject_path = os.path.abspath(os.path.join('..','..','pyproject.toml'))

with open(pyproject_path, 'r') as f:
    pyproject_data = toml.load(f)

project = 'sbm'
copyright = '2024, Yusuke Takase'
author = "Yusuke Takase"
release = pyproject_data['tool']['poetry']['version']

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'nbsphinx',
    'pydata_sphinx_theme'
]

autosummary_generate = True
autosummary_imported_members = True
autosectionlabel_prefix_document = True
autoclass_content = "class"

exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
