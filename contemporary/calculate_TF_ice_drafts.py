import numpy as np
import xarray as xr
import os

#== ice shelf draft ==

bm = xr.open_dataset('/data/njourdain/DATA_ISMIP6/BedMachineAntarctica_2020-07-15_v02_8km.nc')
Zdraft = bm.surface - bm.thickness

#== TF ==

for scenar in ['main','cold','warm','vary']:

    file_in = 'tf_Oyr_contemporary_'+scenar+'_ismip8km_60m_1950-2025.nc' 

    file_out = file_in.replace('tf','tf_ISdraft')
    print(file_out)

    ds = xr.open_dataset(file_in)

    TFdraft = ds.tf.interp(z=Zdraft,method="linear")

    TFdraft.drop_vars('z').to_netcdf(file_out,unlimited_dims="time")

    print('File created !')
