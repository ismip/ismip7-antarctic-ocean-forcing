#!/usr/bin/env python

import os

from i7aof.convert.cmip import convert_cmip

this_dir = os.path.dirname(os.path.abspath(__file__))
config_filename = os.path.join(this_dir, 'test_convert_cmip.cfg')

convert_cmip(
    model='CESM2-WACCM',
    scenario='historical',
    user_config_filename=config_filename,
)
