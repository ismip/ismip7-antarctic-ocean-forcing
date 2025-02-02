"""
Script for creating ISMIP7 Antarctic ocean forcing
"""

import argparse

from i7aof.version import __version__


def main():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-v', '--version',
                        action='version',
                        version=f'ismip6_ocean_forcing {__version__}',
                        help="Show version number and exit")
    args = parser.parse_args()
