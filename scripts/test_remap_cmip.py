#!/usr/bin/env python

import os

from i7aof.remap.cmip import remap_cmip

this_dir = os.path.dirname(os.path.abspath(__file__))
config_filename = os.path.join(this_dir, 'test_remap_cmip.cfg')

remap_cmip(
    model='CESM2-WACCM',
    scenario='historical',
    user_config_filename=config_filename,
)
