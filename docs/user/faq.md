# FAQ

## Where do outputs go?

Outputs are controlled by paths in your config files.

## Do I need to download data manually?

Some downloads are automated (`i7aof.download` submodules); others require credentials or manual steps (TBD).

## Can I write compressed NetCDF output?

Yes. The `i7aof.io.write_netcdf()` function accepts a `compression` argument
similar to `has_fill_values` (bool | dict | callable). When compression is
requested and no engine is specified, the package will use the `h5netcdf`
engine automatically. By default, a sensible `DEFAULT_COMPRESSION` is used
(`zlib=True`, `complevel=4`, `shuffle=True`). You can override these by
providing `compression_opts` or per-variable compression dictionaries.

