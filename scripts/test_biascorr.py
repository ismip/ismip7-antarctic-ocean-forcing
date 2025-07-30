#!/usr/bin/env python
import os

from mpas_tools.config import MpasConfigParser

from i7aof.biascorr.projection import Projection

config = MpasConfigParser()
config.add_from_package('i7aof', 'default.cfg')
config.add_user_config('test_biascorr.cfg')

work_base_dir = config.get('workdir', 'base_dir')
os.makedirs(work_base_dir, exist_ok=True)
os.chdir(work_base_dir)

proj = Projection(config, logger=None)
proj.read_reference()
proj.compute_bias()
proj.read_model()
