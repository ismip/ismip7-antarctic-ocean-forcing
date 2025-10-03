"""Utilities for generating Fortran extrapolation namelists.

This subpackage currently ships a Jinja2 template that can be rendered to
produce a combined horizontal + vertical extrapolation namelist consumed by
the Fortran executables.
"""

from importlib import resources as _resources

__all__ = ['get_template_path', 'load_template_text']


def get_template_path() -> str:
    """Return the filesystem path to the bundled Jinja2 namelist template."""
    template = _resources.files(__package__) / 'namelist_template.nml.j2'
    with _resources.as_file(template) as p:
        return str(p)


def load_template_text() -> str:
    """Return the contents of the bundled Jinja2 namelist template."""
    template = _resources.files(__package__) / 'namelist_template.nml.j2'
    return template.read_text(encoding='utf-8')
