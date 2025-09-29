import os
from typing import List


def get_ct_sa_output_paths(
    config,
    model: str,
    scenario: str,
    workdir: str,
) -> List[str]:
    """
    Derive absolute output paths for converted ct_sa files based on the
    thetao and so file lists in the config. Filenames are formed by replacing
    the variable token (thetao/so) in the thetao basename with 'ct_sa' and
    writing into:

        {workdir}/convert/{model}/{scenario}/Omon/ct_sa/

    Returns a list of absolute output filenames, one per input pair, in the
    same order as the thetao list.

    Parameters
    ----------
    config : mpas_tools.config.MpasConfigParser
        Configuration object containing file lists under
        ``[{scenario}_files]`` for ``thetao`` and ``so``.
    model : str
        Name of the CMIP model (used to build the output directory path).
    scenario : str
        Scenario key (e.g., ``historical``, ``ssp585``) used to select file
        lists from the config.
    workdir : str
        Base working directory where converted files are written under
        ``convert/{model}/{scenario}/Omon/ct_sa``.

    Returns
    -------
    List[str]
        Absolute paths to ct_sa output files, one per thetao/so pair, in the
        same order as the ``thetao`` list.
    """
    thetao_paths = list(config.getexpression(f'{scenario}_files', 'thetao'))
    so_paths = list(config.getexpression(f'{scenario}_files', 'so'))

    if len(thetao_paths) != len(so_paths):
        raise ValueError(
            'Mismatched number of thetao and so files for scenario '
            f"'{scenario}'."
        )

    outdir = os.path.join(workdir, 'convert', model, scenario, 'Omon', 'ct_sa')

    out_paths: List[str] = []
    for th_rel, _ in zip(thetao_paths, so_paths, strict=True):
        th_base = os.path.basename(th_rel)
        if 'thetao' in th_base:
            ct_base = th_base.replace('thetao', 'ct_sa')
        else:
            ct_base = f'ct_sa_{th_base}'
        out_paths.append(os.path.join(outdir, ct_base))

    return out_paths
