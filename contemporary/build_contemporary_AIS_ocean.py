########################################################################
# Builds the netcdf files for the AIS ocean ISMIP7 contemporary period
# for potential temperature, practical salinity and thermal forcing
#
# Nico Jourdain (IGE/CNRS/IRD), January 2026.
#
########################################################################
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

#======================================================================
#-- Define variable attributes:
def add_standard_attributes_oce(ds,miss=9.969209968386869e36):

    ds.time.encoding['units'] = 'days since 1850-01-01'
    ds.time.encoding['_FillValue'] = None
    ds.time.attrs['standard_name'] = 'time'
    ds.time.attrs['long_name'] = 'time'

    ds.x.attrs['units'] = 'm'
    ds.x.encoding['_FillValue'] = None
    ds.x.attrs['long_name'] = 'x coordinate of projection'
    ds.x.attrs['standard_name'] = 'projection_x_coordinate'
    ds.x.attrs['axis'] = 'X'
    ds.x.attrs['bounds'] = 'x_bnds'

    ds.y.attrs['units'] = 'm'
    ds.y.encoding['_FillValue'] = None
    ds.y.attrs['long_name'] = 'y coordinate of projection'
    ds.y.attrs['standard_name'] = 'projection_y_coordinate'
    ds.y.attrs['axis'] = 'Y'
    ds.y.attrs['bounds'] = 'y_bnds'

    ds.z.attrs['units'] = 'm'
    ds.z.encoding['_FillValue'] = None
    ds.z.attrs['long_name'] = 'height relative to sea surface (positive up)'
    ds.z.attrs['standard_name'] = 'height'
    ds.z.attrs['positive'] = 'up'
    ds.z.attrs['axis'] = 'Z'
    ds.z.attrs['bounds'] = 'z_bnds'

    ds.lon.attrs['units'] = 'degrees_east'
    ds.lon.encoding['_FillValue'] = None
    ds.lon.attrs['long_name'] = 'longitude coordinate'
    ds.lon.attrs['standard_name'] = 'longitude'
    ds.lon.attrs['bounds'] = 'lon_bnds'

    ds.lat.attrs['units'] = 'degrees_north'
    ds.lat.encoding['_FillValue'] = None
    ds.lat.attrs['long_name'] = 'latitude coordinate'
    ds.lat.attrs['standard_name'] = 'latitude'
    ds.lat.attrs['bounds'] = 'lat_bnds'

    if ( "thetao" in ds.data_vars ):
        ds.thetao.attrs['_FillValue'] = miss
        ds.thetao.attrs['units'] = 'degC'
        ds.thetao.attrs['long_name'] = 'Sea Water Potential Temperature'
        ds.thetao.attrs['standard_name'] = 'sea_water_potential_temperature'

    if ( "so" in ds.data_vars ):
        ds.so.attrs['_FillValue'] = miss
        ds.so.attrs['units'] = '0.001'
        ds.so.attrs['long_name'] = 'Sea Water Salinity (practical salinity)'
        ds.so.attrs['standard_name'] = 'sea_water_salinity'

    if ( "tf" in ds.data_vars ):
        ds.tf.attrs['_FillValue'] = miss
        ds.tf.attrs['units'] = 'degC'
        ds.tf.attrs['long_name'] = 'Thermal Forcing'
        ds.tf.attrs['standard_name'] = 'thermal_forcing'

    return ds


#======================================================================
#-- Calculate 3d thermal forcing and define as xarray dataset:
def calculate_TF(T,S):
    lbd1 = -0.0573  # [degC PSU^-1] Liquidus slope
    lbd2 =  0.0832  # [degC]        Liquidus intercept
    lbd3 = -7.53e-8  # [degC Pa^-1]  Liquidus pressure coefficient
    g     = 9.81    # [m s^-2]
    rhosw = 1028.   # [kg m^-3]

    Tzd = lbd1 * S + lbd2 - lbd3 * g * rhosw * dT.z
    TF = T - Tzd
    return TF

#======================================================================

print('Starting...')
file_T = 'OI_Climatology_ismip8km_60m_thetao_extrap.nc'
file_S = 'OI_Climatology_ismip8km_60m_so_extrap.nc'
dT = xr.open_dataset(file_T)
dS = xr.open_dataset(file_S)
Tclim = np.float32(dT.thetao.values)
Sclim = np.float32(dS.so.values)

file_basin = 'imbie2_basin_numbers_8km_v2.nc'
dB = xr.open_dataset(file_basin)
msk_AUR = np.float32(dB.basinNumber.where((dB.basinNumber==4),0).values)/4.
msk_AMU = np.float32(dB.basinNumber.where((dB.basinNumber==9),0).values)/9.
msk_GET = np.float32(dB.basinNumber.where((dB.basinNumber==8),0).values)/8.

years = np.arange(1950,2026,1)
mt = np.size(years)
mz,my,mx = dT.thetao.shape

scenar = ['main','cold','warm','vary']
scenar_long=['main plausible pathway','cold plausible pathway','warm plausible pathway','varying plausible pathway']
for ks in range(len(scenar)):

    print(scenar_long[ks])

    T_xxxx = np.zeros((mt,mz,my,mx),dtype='float32')
    S_xxxx = np.zeros((mt,mz,my,mx),dtype='float32')

    ## General method (not Amundsen or Aurora):
    for kt in range(mt):
       T_xxxx[kt,:,:,:] = Tclim
       S_xxxx[kt,:,:,:] = Sclim
    if scenar[ks] == 'cold':
       T_xxxx = T_xxxx - 0.17
    elif scenar[ks] == 'warm':
       T_xxxx = T_xxxx + 0.17

    ## Eastern Amundsen Sea & Getz
    if (scenar[ks] == 'main'):
        fac = -0.00140
    elif (scenar[ks] == 'cold'):
        fac = -0.00620
    elif (scenar[ks] == 'warm'):
        fac =  0.00190
    else:
        fac =  0.00000

    if scenar[ks] != 'vary':
        for yr in np.arange(years[0],1995,1):
            kt=yr-years[0]
            kt1995=1995-years[0]
            for kz in range(mz):
                T_xxxx[kt,kz,:,:] = T_xxxx[kt1995,kz,:,:] + fac * (kt1995-kt) * ( 0.5 * msk_GET[:,:] + 1.0 * msk_AMU[:,:] )

    ## Aurora basin (only the coldest pathway has a specific treatment)
    if scenar[ks] == 'cold':
        kt2015=2015-years[0]
        meanSclim = np.mean(Sclim[0:13,:,:],axis=0) # mean salinity in the first ~700m
        for yr in np.arange(years[0],2015,1):
            kt=yr-years[0]
            for kz in range(mz):
                T_xxxx[kt,kz,:,:] = T_xxxx[kt,kz,:,:] * ( 1 - msk_AUR[:,:] ) \
                                    + msk_AUR[:,:] * ( -1.9 + ( T_xxxx[kt2015,kz,:,:] - (-1.9) ) * kt / (2015-years[0]) )
                S_xxxx[kt,kz,:,:] = S_xxxx[kt,kz,:,:] * ( 1 - msk_AUR[:,:] ) \
                                    + msk_AUR[:,:] * ( meanSclim[:,:] + ( S_xxxx[kt2015,kz,:,:] - meanSclim[:,:] ) * kt / (2015-years[0]) )

    ## Varying plausible pathway
    if scenar[ks] == 'vary':
        zz=dT.z.values
        wave = np.cos(2*np.pi*(years-2008)/16) # min in 2000, max in 2008, 16-year period.
        delta_dep_thcl_AMU = 120.0 * wave      # Dutrieux et al. (2014)
        delta_dep_thcl_GET = 100.0 * wave      # Jacobs et al. (2013)
        delta_dep_thcl_AUR = 100.0 * wave      # Hirano et al. (2023)
        for kt in range(mt):
            dz_AMU = delta_dep_thcl_AMU[kt]
            dz_GET = delta_dep_thcl_GET[kt]
            dz_AUR = delta_dep_thcl_AUR[kt]
            print('kt = ', kt)
            for kz in range(mz):
                kzinf_AMU = np.amax([np.amin([kz+np.ceil(dz_AMU/60).astype('int'), mz-1]),1])
                kzinf_GET = np.amax([np.amin([kz+np.ceil(dz_GET/60).astype('int'), mz-1]),1])
                kzinf_AUR = np.amax([np.amin([kz+np.ceil(dz_AUR/60).astype('int'), mz-1]),1])
                kzsup_AMU = kzinf_AMU-1
                kzsup_GET = kzinf_GET-1
                kzsup_AUR = kzinf_AUR-1
                zintrp_AMU = np.amax([np.amin([zz[kz]-dz_AMU,zz[0]]),zz[-1]])
                zintrp_GET = np.amax([np.amin([zz[kz]-dz_GET,zz[0]]),zz[-1]])
                zintrp_AUR = np.amax([np.amin([zz[kz]-dz_AUR,zz[0]]),zz[-1]])
                if kt < 17:
                   print('    ',kz, kzinf_AMU, kzsup_AMU, zintrp_AMU, zz[kzinf_AMU], zz[kzsup_AMU])
                T_xxxx[kt,kz,:,:] =   msk_AMU[:,:] * (   (zz[kzsup_AMU]-zintrp_AMU) * Tclim[kzinf_AMU,:,:] \
                                                       + (zintrp_AMU-zz[kzinf_AMU]) * Tclim[kzsup_AMU,:,:] ) / (zz[kzsup_AMU]-zz[kzinf_AMU]+1.e-12) \
                                    + msk_GET[:,:] * (   (zz[kzsup_GET]-zintrp_GET) * Tclim[kzinf_GET,:,:] \
                                                       + (zintrp_GET-zz[kzinf_GET]) * Tclim[kzsup_GET,:,:] ) / (zz[kzsup_GET]-zz[kzinf_GET]+1.e-12) \
                                    + msk_AUR[:,:] * (   (zz[kzsup_AUR]-zintrp_AUR) * Tclim[kzinf_AUR,:,:] \
                                                       + (zintrp_AUR-zz[kzinf_AUR]) * Tclim[kzsup_AUR,:,:] ) / (zz[kzsup_AUR]-zz[kzinf_AUR]+1.e-12) \
                                    + ( 1 - msk_AMU[:,:] - msk_GET[:,:] - msk_AUR[:,:] ) * Tclim[kz,:,:]
                S_xxxx[kt,kz,:,:] =   msk_AMU[:,:] * (   (zz[kzsup_AMU]-zintrp_AMU) * Sclim[kzinf_AMU,:,:] \
                                                       + (zintrp_AMU-zz[kzinf_AMU]) * Sclim[kzsup_AMU,:,:] ) / (zz[kzsup_AMU]-zz[kzinf_AMU]+1.e-12) \
                                    + msk_GET[:,:] * (   (zz[kzsup_GET]-zintrp_GET) * Sclim[kzinf_GET,:,:] \
                                                       + (zintrp_GET-zz[kzinf_GET]) * Sclim[kzsup_GET,:,:] ) / (zz[kzsup_GET]-zz[kzinf_GET]+1.e-12) \
                                    + msk_AUR[:,:] * (   (zz[kzsup_AUR]-zintrp_AUR) * Sclim[kzinf_AUR,:,:] \
                                                       + (zintrp_AUR-zz[kzinf_AUR]) * Sclim[kzsup_AUR,:,:] ) / (zz[kzsup_AUR]-zz[kzinf_AUR]+1.e-12) \
                                    + ( 1 - msk_AMU[:,:] - msk_GET[:,:] - msk_AUR[:,:] ) * Sclim[kz,:,:]

    ## Save T,S,TF as xarray data arrays

    #-- Define time:
    yr_st=years.astype('str')
    for kt in range(mt):
        yr_st[kt]=yr_st[kt]+'-07-01T12:00:00'
    time_val = np.array(yr_st,dtype='datetime64[s]')

    print('    Creating xarray datasets for T,S...')
    #-- Create xarray datasets for T,S:
    varlong=['thetao','so']
    var=['T','S']
    dTout_xxxx = xr.Dataset( {  "thetao": (["time", "z", "y", "x"],np.float32(T_xxxx)),
                                "lat": (["y","x"],np.float64(dT.lat)), "lon": (["y","x"],np.float64(dT.lon)),
                                "lat_bnds": (["y","x","nv"],np.float64(dT.lat_bnds)), "lon_bnds": (["y","x","nv"],np.float64(dT.lon_bnds)),
                                "x_bnds": (["x","bnds"],np.float64(dT.x_bnds)), "y_bnds": (["y","bnds"],np.float64(dT.y_bnds)), "z_bnds": (["z","bnds"],np.float64(dT.z_bnds))  },
                             coords={  "time": time_val, "z": np.float64(dT.z.values), "y": np.float64(dT.y.values), "x": np.float64(dT.x.values) },
                             attrs={  "history":"created using build_contemporary_AIS_ocean.py",
                                      "project": "ISMIP7 AIS ocean contemporary forcing",
                                      "scenario": scenar_long[ks]  } )
    dSout_xxxx = xr.Dataset( {  "so": (["time", "z", "y", "x"],np.float32(S_xxxx)),
                                "lat": (["y","x"],np.float64(dT.lat)), "lon": (["y","x"],np.float64(dT.lon)),
                                "lat_bnds": (["y","x","nv"],np.float64(dT.lat_bnds)), "lon_bnds": (["y","x","nv"],np.float64(dT.lon_bnds)),
                                "x_bnds": (["x","bnds"],np.float64(dT.x_bnds)), "y_bnds": (["y","bnds"],np.float64(dT.y_bnds)), "z_bnds": (["z","bnds"],np.float64(dT.z_bnds))  },
                             coords={  "time": time_val, "z": np.float64(dT.z.values), "y": np.float64(dT.y.values), "x": np.float64(dT.x.values) },
                             attrs={  "history":"created using build_contemporary_AIS_ocean.py",
                                      "project": "ISMIP7 AIS ocean contemporary forcing",
                                      "scenario": scenar_long[ks]  } )

    print('    Calculating 3d thermal forcing and define as xarray dataset...')
    TF_xxxx = calculate_TF(dTout_xxxx.thetao,dSout_xxxx.so)
    dTFout_xxxx = xr.Dataset( {  "tf": (["time", "z", "y", "x"],np.float32(TF_xxxx)),
                                 "lat": (["y","x"],np.float64(dT.lat)), "lon": (["y","x"],np.float64(dT.lon)),
                                 "lat_bnds": (["y","x","nv"],np.float64(dT.lat_bnds)), "lon_bnds": (["y","x","nv"],np.float64(dT.lon_bnds)),
                                 "x_bnds": (["x","bnds"],np.float64(dT.x_bnds)), "y_bnds": (["y","bnds"],np.float64(dT.y_bnds)), "z_bnds": (["z","bnds"],np.float64(dT.z_bnds))  },
                             coords={  "time": time_val, "z": np.float64(dT.z.values), "y": np.float64(dT.y.values), "x": np.float64(dT.x.values) },
                             attrs={  "history":"created using build_contemporary_AIS_ocean.py",
                                      "project": "ISMIP7 AIS ocean contemporary forcing",
                                      "scenario": scenar_long[ks]  } )

    print('    Defining variable attributes...')
    add_standard_attributes_oce(dTout_xxxx)
    add_standard_attributes_oce(dSout_xxxx)
    add_standard_attributes_oce(dTFout_xxxx)
    file_out_T  = 'thetao_Oyr_contemporary_'+scenar[ks]+'_ismip8km_60m_1950-2025.nc'
    file_out_S  = 'so_Oyr_contemporary_'+scenar[ks]+'_ismip8km_60m_1950-2025.nc'
    file_out_TF = 'tf_Oyr_contemporary_'+scenar[ks]+'_ismip8km_60m_1950-2025.nc'
    print('    Creating',file_out_T)
    dTout_xxxx.to_netcdf(  file_out_T  , unlimited_dims="time")
    print('    Creating',file_out_S)
    dSout_xxxx.to_netcdf(  file_out_S  , unlimited_dims="time")
    print('    Creating',file_out_TF)
    dTFout_xxxx.to_netcdf( file_out_TF , unlimited_dims="time")

print('[oK]')
