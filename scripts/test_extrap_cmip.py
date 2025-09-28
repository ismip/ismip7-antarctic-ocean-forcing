#!/usr/bin/env python

import os

from i7aof.extrap.cmip import extrap_cmip

this_dir = os.path.dirname(os.path.abspath(__file__))
config_filename = os.path.join(this_dir, 'test_extrap_cmip.cfg')

# Assumes the convert and remap workflows have already been run in the same
# work directory referenced by test_extrap_cmip.cfg.
extrap_cmip(
    model='CESM2-WACCM',
    scenario='historical',
    user_config_filename=config_filename,
)
