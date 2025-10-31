from mpas_tools.config import MpasConfigParser

from i7aof.cmip import get_model_prefix


def load_config(
    model: str | None = None,
    clim_name: str | None = None,
    inputdir: str | None = None,
    workdir: str | None = None,
    user_config_filename: str | None = None,
) -> MpasConfigParser:
    """
    Convert thetao/so monthly files to ct/sa for a CMIP model.

    Conversion is performed per thetao/so pair. Outputs store only ct & sa
    plus coordinate variables; bounds variables for depth/lat/lon are
    injected only in the final merged file to avoid acquiring a spurious
    time dimension.

    Parameters
    ----------
    model : str
        Name of the CMIP model (used to select the model config and to
        construct output paths).
    clim_name: str
        The name of the reference climatology (e.g. "zhou_annual_30_sep.cfg").
    inputdir : str, optional
        Base directory where the relative input file paths are resolved. If
        not provided, uses ``[inputdir] base_dir`` from the config.
    workdir : str, optional
        Base working directory where outputs will be written. If not
        provided, uses ``[workdir] base_dir`` from the config.
    user_config_filename : str, optional
        Optional user config that overrides defaults (paths, variable names,
        chunk sizes, etc.).
    """

    config = MpasConfigParser()
    config.add_from_package('i7aof', 'default.cfg')
    if model is not None:
        config.add_from_package('i7aof.cmip', f'{get_model_prefix(model)}.cfg')
    if clim_name is not None:
        config.add_from_package('i7aof.clim', f'{clim_name}.cfg')
    if user_config_filename is not None:
        config.add_user_config(user_config_filename)

    if workdir is not None:
        config.set('workdir', 'base_dir', workdir)

    if inputdir is not None:
        config.set('inputdir', 'base_dir', inputdir)

    if not config.has_option('workdir', 'base_dir'):
        raise ValueError(
            'Missing configuration option: [workdir] base_dir. '
            'Please supply a user config file or command-line option that '
            'defines this option.'
        )

    return config


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _get_or_config_path(
    config: MpasConfigParser, supplied: str | None, section: str
) -> str:
    if supplied is not None:
        return supplied
    if config.has_option(section, 'base_dir'):
        return config.get(section, 'base_dir')
    raise ValueError(
        f'Missing configuration option: [{section}] base_dir. '
        'Please supply a user config file that defines this option.'
    )
