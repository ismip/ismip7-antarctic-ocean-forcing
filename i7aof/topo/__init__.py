from i7aof.topo.bedmachine import BedMachineAntarcticaV3
from i7aof.topo.bedmap import Bedmap3


def get_topo(config, logger):
    """
    Get the topography object.

    Parameters
    ----------
    config : mpas_tools.config.MpasConfigParser
        Configuration options.

    logger : logging.Logger
        Logger for output from building mapping files and remapping datasets

    Returns
    -------
    TopoBase
        The topography object.
    """
    topo_dataset = config.get('topo', 'dataset')
    if topo_dataset == 'bedmachine_antarctica_v3':
        topo = BedMachineAntarcticaV3(config, logger)
    elif topo_dataset == 'bedmap3':
        topo = Bedmap3(config, logger)
    else:
        raise ValueError(f'Unknown topography dataset: {topo_dataset}')
    return topo
