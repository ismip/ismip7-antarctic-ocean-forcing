# Configuration file for the Sphinx documentation builder.
from __future__ import annotations
import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------
# Add project root to sys.path for autodoc if needed
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'ISMIP7 Antarctic Ocean Forcing (i7aof)'
author = 'ISMIP contributors'
current_year = datetime.now().year
copyright = f'{current_year}, {author}'

# Import version from the package
try:
    from i7aof.version import __version__ as release
except Exception:
    release = '0.0.0-dev'

# -- General configuration ---------------------------------------------------
extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_copybutton',
    'sphinx_design',
]

myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'linkify',
    'substitution',
]

# The master toctree document.
master_doc = 'index'

# Templates path
templates_path = ['_templates']

# Patterns to ignore when looking for source files.
exclude_patterns: list[str] = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_static_path = ['_static']
html_title = f'i7aof {release}'

# -- MyST substitutions ------------------------------------------------------
myst_substitutions = {
    'package': 'i7aof',
    'project': 'ismip7-antarctic-ocean-forcing',
}
