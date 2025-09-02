# Quickstart

This quickstart shows how to produce a small test forcing using built-in scripts.

## Command-line interfaces

Two entry points are provided:

- `ismip7-antarctic-ocean-forcing` — main workflow driver (under development).
- `ismip7-antarctic-remap-cmip` — remap CMIP data to the ISMIP grid.

### Example: run remapping with a test config

```bash
ismip7-antarctic-remap-cmip \
    --model CESM2-WACCM \
    --variable thetao \
    --scenario historical \
    --config scripts/test_remap_cmip.cfg
```

### Example: run bias correction test

```python
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
```

Notes:
- Use `--help` on any module for options.
- Outputs will be written according to your config files.
