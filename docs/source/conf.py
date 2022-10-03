# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PresetAid'
copyright = '2022, Grzegorz Krug'
author = 'Grzegorz Krug'
release = '0.1'

import os
import sys


# -- PATH System Edit
#sys.path.append(os.path.abspath("..")) # add workdir
# main directory of repo
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..')) # add parent dir docs/source/../.. 


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.todo', 'sphinx.ext.autosummary']

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'haiku'
html_theme = 'nature'
html_theme = 'pyramid'

html_static_path = ['_static']

html_theme_options = {
    'sidebarwidth' : '300px',
    'body_max_width' : '1000px',
    #'linkcolor' : 'RGB(0,190,150)',
}

def setup(app):
    app.add_css_file('custom.css')  # may also be an URL

