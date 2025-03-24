"""
Script for creating ISMIP7 Antarctic ocean forcing
"""

import argparse

from mpas_tools.config import MpasConfigParser

from i7aof.version import __version__


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-c', '--config', nargs='*', type=str, help='Configuration file(s)'
    )
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=f'ismip6_ocean_forcing {__version__}',
        help='Show version number and exit',
    )
    args = parser.parse_args()

    config = MpasConfigParser()
    config.add_from_package('i7aof', 'default.cfg')
    for config_file in args.config:
        config.add_user_config(config_file)
