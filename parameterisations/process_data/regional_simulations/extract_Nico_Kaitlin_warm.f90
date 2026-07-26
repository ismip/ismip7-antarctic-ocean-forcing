program modif
 
USE netcdf
 
IMPLICIT NONE
 
INTEGER :: fidMITgcm, status, dimID_x, dimID_y, dimID_z, dimID_time, mx, my, mz, mtime, mltMITgcm_ID,   &
&          salMITgcm_ID, temMITgcm_ID, x_ID, y_ID, z_ID, time_ID, fidm, basinNumber_ID, fidB, N_MITgcm, &
&          fidNEMO, salNEMO_ID, temNEMO_ID, mltNEMO_ID, sftflf_ID, kk, N_NEMO, i, j, fidS, fidT,        &
&          fidSclim, fidTclim, dimID_bnds, mbnds, z_bnds_ID, y_bnds_ID, x_bnds_ID, sal_ID, tem_ID, year,&
&          mlt_ID, mlt_std_ID, yeari_MITgcm, yearf_MITgcm, fidBM, rock_frac_ID, ice_frac_ID, sal_std_ID,&
&          tem_std_ID
 
CHARACTER(LEN=1) :: exprt

CHARACTER(LEN=180) :: file_in_MITgcm, file_in_NEMO, file_in_B, file_out_S, file_out_T, file_out_m, &
&                     file_in_Sclim, file_in_Tclim, file_in_BM

CHARACTER(LEN=300) :: desc

INTEGER*4,ALLOCATABLE,DIMENSION(:,:) :: basinNumber
 
REAL*8,ALLOCATABLE,DIMENSION(:) :: x, y, z

REAL*8,ALLOCATABLE,DIMENSION(:,:) :: z_bnds, y_bnds, x_bnds
 
REAL*4,ALLOCATABLE,DIMENSION(:,:) :: mltMITgcm, txp, typ, mltNEMO, sftflf, mltMITgcm_std, mltNEMO_std,    &
&                                    mlt, mlt_std, tzp, rock_frac, ice_frac
 
REAL*4,ALLOCATABLE,DIMENSION(:,:,:) :: salMITgcm, temMITgcm, salMITgcm_std, temMITgcm_std, tmp, sal, tem, &
&                                      salNEMO, temNEMO, salNEMO_std, temNEMO_std, sal_std, tem_std

REAL*4 :: miss

!---------------------------------------

file_out_S = 'Jourdain-Naughten_NEMO-MITgcm_warm_S.nc'
file_out_T = 'Jourdain-Naughten_NEMO-MITgcm_warm_T.nc'
file_out_m = 'Jourdain-Naughten_NEMO-MITgcm_warm_m.nc'

file_in_B  = '/scratchu/njourdain/ISMIP7_REGIONAL_OCEAN_DATA/DATA/basin_numbers_ismip8km_v2.nc'
file_in_BM  = '/scratchu/njourdain/ISMIP7_REGIONAL_OCEAN_DATA/DATA/BedMachineAntarctica-v3_AIS_obs_ocean_topography_v3.nc'
file_in_Sclim  = '/scratchu/njourdain/ISMIP7_REGIONAL_OCEAN_DATA/DATA/so_AIS_obs_ocean_climatology_zhou_annual_06_nov_v4_1972-2024.nc'
file_in_Tclim  = '/scratchu/njourdain/ISMIP7_REGIONAL_OCEAN_DATA/DATA/thetao_AIS_obs_ocean_climatology_zhou_annual_06_nov_v4_1972-2024.nc'

101 FORMAT('/scratchu/njourdain/ISMIP7_REGIONAL_OCEAN_DATA/DATA/Kaitlin/WED/MITgcm_WS_1pctCO2_',i4,'.nc')
102 FORMAT('/scratchu/njourdain/ISMIP7_REGIONAL_OCEAN_DATA/DATA/Nico/climato_Amundsen_NEMO-',a,'_2080-2100_rcp85.nc')

yeari_MITgcm = 2030
yearf_MITgcm = 2049
mz = 30

111 FORMAT("Merge of MITgcm-Weddell-1pctCO2 over ",i4,"-",i4," (Naughten 2021) in basins 0, 12, 13, 14, 15 and of NEMO-Amundsen-hist (for A, B, C exp. of Jourdain 2022) in basins 8, 9, 10; climatology elsewhere")
write(desc,111) yeari_MITgcm, yearf_MITgcm
 
miss = 1.e20

!---------------------------------------
! Read basins:

write(*,*) 'Reading ', TRIM(file_in_B)

status = NF90_OPEN(TRIM(file_in_B),0,fidB); call erreur(status,.TRUE.,"read")

status = NF90_INQ_DIMID(fidB,"x",dimID_x); call erreur(status,.TRUE.,"inq_dimID_x")
status = NF90_INQ_DIMID(fidB,"y",dimID_y); call erreur(status,.TRUE.,"inq_dimID_y")

status = NF90_INQUIRE_DIMENSION(fidB,dimID_x,len=mx); call erreur(status,.TRUE.,"inq_dim_x")
status = NF90_INQUIRE_DIMENSION(fidB,dimID_y,len=my); call erreur(status,.TRUE.,"inq_dim_y")

ALLOCATE(  y(my)  ) 
ALLOCATE(  x(mx)  )
ALLOCATE(  basinNumber(mx,my)  )
 
status = NF90_INQ_VARID(fidB,"y",y_ID); call erreur(status,.TRUE.,"inq_y_ID")
status = NF90_INQ_VARID(fidB,"x",x_ID); call erreur(status,.TRUE.,"inq_x_ID")
status = NF90_INQ_VARID(fidB,"basinNumber",basinNumber_ID); call erreur(status,.TRUE.,"inq_basinNumber_ID")
 
status = NF90_GET_VAR(fidB,y_ID,y); call erreur(status,.TRUE.,"getvar_y")
status = NF90_GET_VAR(fidB,x_ID,x); call erreur(status,.TRUE.,"getvar_x")
status = NF90_GET_VAR(fidB,basinNumber_ID,basinNumber); call erreur(status,.TRUE.,"getvar_basinNumber")

status = NF90_CLOSE(fidB); call erreur(status,.TRUE.,"close_file")

!---------------------------------------
! Read mask
 
write(*,*) 'Reading ', TRIM(file_in_BM)
 
status = NF90_OPEN(TRIM(file_in_BM),0,fidBM); call erreur(status,.TRUE.,"read")
 
ALLOCATE(  rock_frac(mx,my)  ) 
ALLOCATE(  ice_frac(mx,my)  ) 
 
status = NF90_INQ_VARID(fidBM,"rock_frac",rock_frac_ID); call erreur(status,.TRUE.,"inq_rock_frac_ID")
status = NF90_INQ_VARID(fidBM,"ice_frac",ice_frac_ID); call erreur(status,.TRUE.,"inq_ice_frac_ID")
 
status = NF90_GET_VAR(fidBM,rock_frac_ID,rock_frac); call erreur(status,.TRUE.,"getvar_rock_frac")
status = NF90_GET_VAR(fidBM,ice_frac_ID,ice_frac); call erreur(status,.TRUE.,"getvar_ice_frac")
 
status = NF90_CLOSE(fidBM); call erreur(status,.TRUE.,"close_file")

ice_frac = ice_frac + rock_frac
DEALLOCATE( rock_frac )

!-----
ALLOCATE(  z(mz)  )

!=======================================================================================================
! SALINITY
!=======================================================================================================

ALLOCATE(  sal(mx,my,mz), sal_std(mx,my,mz)  )
ALLOCATE(  salMITgcm(mx,my,mz)  )
ALLOCATE(  salMITgcm_std(mx,my,mz)  )
ALLOCATE(  salNEMO(mx,my,mz)  )
ALLOCATE(  salNEMO_std(mx,my,mz)  )
ALLOCATE(  tmp(mx,my,mz)  )

!---------------------------------------
! Read salinity climatology
 
write(*,*) 'Reading ', TRIM(file_in_Sclim)
 
status = NF90_OPEN(TRIM(file_in_Sclim),0,fidSclim); call erreur(status,.TRUE.,"read")
 
status = NF90_INQ_DIMID(fidSclim,"bnds",dimID_bnds); call erreur(status,.TRUE.,"inq_dimID_bnds")
 
status = NF90_INQUIRE_DIMENSION(fidSclim,dimID_bnds,len=mbnds); call erreur(status,.TRUE.,"inq_dim_bnds")
  
ALLOCATE(  z_bnds(mbnds,mz)  ) 
ALLOCATE(  y_bnds(mbnds,my)  ) 
ALLOCATE(  x_bnds(mbnds,mx)  ) 
 
status = NF90_INQ_VARID(fidSclim,"z_bnds",z_bnds_ID); call erreur(status,.TRUE.,"inq_z_bnds_ID")
status = NF90_INQ_VARID(fidSclim,"y_bnds",y_bnds_ID); call erreur(status,.TRUE.,"inq_y_bnds_ID")
status = NF90_INQ_VARID(fidSclim,"x_bnds",x_bnds_ID); call erreur(status,.TRUE.,"inq_x_bnds_ID")
status = NF90_INQ_VARID(fidSclim,"so",sal_ID); call erreur(status,.TRUE.,"inq_sal_ID")
 
status = NF90_GET_VAR(fidSclim,z_bnds_ID,z_bnds); call erreur(status,.TRUE.,"getvar_z_bnds")
status = NF90_GET_VAR(fidSclim,y_bnds_ID,y_bnds); call erreur(status,.TRUE.,"getvar_y_bnds")
status = NF90_GET_VAR(fidSclim,x_bnds_ID,x_bnds); call erreur(status,.TRUE.,"getvar_x_bnds")
status = NF90_GET_VAR(fidSclim,sal_ID,sal); call erreur(status,.TRUE.,"getvar_so")
 
status = NF90_CLOSE(fidSclim); call erreur(status,.TRUE.,"close_file")

!---------------------------------------
! Read MITgcm data:

N_MITgcm = 0
salMITgcm = 0.e0
salMITgcm_std = 0.e0

do year=yeari_MITgcm,yearf_MITgcm

  write(file_in_MITgcm,101) year 

  write(*,*) 'Reading ', TRIM(file_in_MITgcm)
   
  status = NF90_OPEN(TRIM(file_in_MITgcm),0,fidMITgcm); call erreur(status,.TRUE.,"read")
   
  status = NF90_INQ_VARID(fidMITgcm,"salinity",salMITgcm_ID); call erreur(status,.TRUE.,"inq_salMITgcm_ID")
  status = NF90_INQ_VARID(fidMITgcm,"z",z_ID); call erreur(status,.TRUE.,"inq_z_ID")
   
  status = NF90_GET_VAR(fidMITgcm,salMITgcm_ID,tmp,start=(/1,1,2/),count=(/mx,my,mz/),stride=(/1,1,3/)); call erreur(status,.TRUE.,"getvar_salMITgcm")
  salMITgcm = salMITgcm + tmp
  salMITgcm_std = salMITgcm_std + tmp**2
  status = NF90_GET_VAR(fidMITgcm,z_ID,z,start=(/2/),count=(/mz/),stride=(/3/)); call erreur(status,.TRUE.,"getvar_z")
   
  status = NF90_CLOSE(fidMITgcm); call erreur(status,.TRUE.,"close_file")

  N_MITgcm = N_MITgcm + 1

enddo

salMITgcm = salMITgcm / N_MITgcm
salMITgcm_std = salMITgcm_std / N_MITgcm
salMITgcm_std = ( salMITgcm_std - salMITgcm**2 )**0.5

!---------------------------------------
! Read Nico's Amundsen

salNEMO = 0.e0
salNEMO_std = 0.e0
N_NEMO = 0

do kk=1,3

  if ( kk .eq. 1 ) then
     exprt='A'
  elseif ( kk .eq. 2 ) then
     exprt='B'
  elseif ( kk .eq. 3 ) then
     exprt='C' 
  endif

  write(file_in_NEMO,102) exprt
 
  write(*,*) 'Reading ', TRIM(file_in_NEMO)
   
  status = NF90_OPEN(TRIM(file_in_NEMO),0,fidNEMO); call erreur(status,.TRUE.,"read")
   
  status = NF90_INQ_VARID(fidNEMO,"so",salNEMO_ID); call erreur(status,.TRUE.,"inq_salNEMO_ID")
   
  status = NF90_GET_VAR(fidNEMO,salNEMO_ID,tmp); call erreur(status,.TRUE.,"getvar_salNEMO")
  salNEMO = salNEMO + tmp
  salNEMO_std = salNEMO_std + tmp**2
   
  status = NF90_CLOSE(fidNEMO); call erreur(status,.TRUE.,"close_file")

  N_NEMO = N_NEMO + 1

enddo

salNEMO = salNEMO / N_NEMO
salNEMO_std = salNEMO_std / N_NEMO
salNEMO_std = ( salNEMO_std - salNEMO**2 )**0.5

!---------------------------------------
! Replace climatology with model outputs in some basins

do i=1,mx
do j=1,my

  if ( basinNumber(i,j) .ge. 12 ) then ! Naughten (not taking basin 0 as Beaudouin is missing in the basin)
    sal(i,j,:) = salMITgcm(i,j,:)
    sal_std(i,j,:) = salMITgcm_std(i,j,:)
  elseif ( basinNumber(i,j) .ge. 8 .and. basinNumber(i,j) .le. 10 ) then ! Jourdain
    sal(i,j,:) = salNEMO(i,j,:)
    sal_std(i,j,:) = salNEMO_std(i,j,:)
  endif

  do kk=1,mz
    if ( isnan(sal(i,j,kk)) .or. sal(i,j,kk) .lt. 0.0 .or. sal(i,j,kk) .gt. 75 ) then
      sal(i,j,kk) = miss
    endif
    if ( isnan(sal_std(i,j,kk)) .or. sal(i,j,kk) .lt. 0.0 .or. sal(i,j,kk) .gt. 75 ) then
      sal_std(i,j,kk) = miss
    endif
  enddo

  ! masking ice shelf cavities and grounded ice
  if ( ice_frac(i,j) .gt. 0.5 ) then
    sal(i,j,:) = miss
    sal_std(i,j,:) = miss
  endif

enddo
enddo 

DEALLOCATE( salMITgcm, salMITgcm_std, salNEMO, salNEMO_std, tmp )

!---------------------------------------
! Writing new netcdf file :
 
write(*,*) 'Creating ', TRIM(file_out_S)
 
status = NF90_CREATE(TRIM(file_out_S),NF90_NOCLOBBER,fidS); call erreur(status,.TRUE.,'create')
 
status = NF90_DEF_DIM(fidS,"x",mx,dimID_x); call erreur(status,.TRUE.,"def_dimID_x")
status = NF90_DEF_DIM(fidS,"y",my,dimID_y); call erreur(status,.TRUE.,"def_dimID_y")
status = NF90_DEF_DIM(fidS,"z",mz,dimID_z); call erreur(status,.TRUE.,"def_dimID_z")
status = NF90_DEF_DIM(fidS,"bnds",mbnds,dimID_bnds); call erreur(status,.TRUE.,"def_dimID_bnds")
  
status = NF90_DEF_VAR(fidS,"so",NF90_FLOAT,(/dimID_x,dimID_y,dimID_z/),sal_ID); call erreur(status,.TRUE.,"def_var_salMITgcm_ID")
status = NF90_DEF_VAR(fidS,"so_uncert",NF90_FLOAT,(/dimID_x,dimID_y,dimID_z/),sal_std_ID); call erreur(status,.TRUE.,"def_var_salMITgcm_ID")
status = NF90_DEF_VAR(fidS,"x",NF90_DOUBLE,(/dimID_x/),x_ID); call erreur(status,.TRUE.,"def_var_x_ID")
status = NF90_DEF_VAR(fidS,"y",NF90_DOUBLE,(/dimID_y/),y_ID); call erreur(status,.TRUE.,"def_var_y_ID")
status = NF90_DEF_VAR(fidS,"z",NF90_DOUBLE,(/dimID_z/),z_ID); call erreur(status,.TRUE.,"def_var_z_ID")
status = NF90_DEF_VAR(fidS,"x_bnds",NF90_DOUBLE,(/dimID_bnds,dimID_x/),x_bnds_ID); call erreur(status,.TRUE.,"def_var_x_bnds_ID")
status = NF90_DEF_VAR(fidS,"y_bnds",NF90_DOUBLE,(/dimID_bnds,dimID_y/),y_bnds_ID); call erreur(status,.TRUE.,"def_var_y_bnds_ID")
status = NF90_DEF_VAR(fidS,"z_bnds",NF90_DOUBLE,(/dimID_bnds,dimID_z/),z_bnds_ID); call erreur(status,.TRUE.,"def_var_z_bnds_ID")
 
status = NF90_PUT_ATT(fidS,sal_ID,"units","1.e-3"); call erreur(status,.TRUE.,"put_att_sal_ID")
status = NF90_PUT_ATT(fidS,sal_ID,"long_name","Sea Water Practical Salinity"); call erreur(status,.TRUE.,"put_att_sal_ID")
status = NF90_PUT_ATT(fidS,sal_ID,"_FillValue",miss); call erreur(status,.TRUE.,"put_att_sal_ID")
status = NF90_PUT_ATT(fidS,sal_std_ID,"units","1.e-3"); call erreur(status,.TRUE.,"put_att_sal_std_ID")
status = NF90_PUT_ATT(fidS,sal_std_ID,"long_name","Sea Water Practical Salinity standard deviation"); call erreur(status,.TRUE.,"put_att_sal_std_ID")
status = NF90_PUT_ATT(fidS,sal_std_ID,"comment","Standard deviation over 20 years for MITgcm and over three 21-year means for NEMO"); call erreur(status,.TRUE.,"put_att_sal_std_ID")
status = NF90_PUT_ATT(fidS,sal_std_ID,"_FillValue",miss); call erreur(status,.TRUE.,"put_att_sal_std_ID")
status = NF90_PUT_ATT(fidS,z_ID,"bounds","z_bnds"); call erreur(status,.TRUE.,"put_att_z_ID")
status = NF90_PUT_ATT(fidS,z_ID,"axis","Z"); call erreur(status,.TRUE.,"put_att_z_ID")
status = NF90_PUT_ATT(fidS,z_ID,"positive","up"); call erreur(status,.TRUE.,"put_att_z_ID")
status = NF90_PUT_ATT(fidS,z_ID,"long_name","height relative to sea surface (positive up)"); call erreur(status,.TRUE.,"put_att_z_ID")
status = NF90_PUT_ATT(fidS,z_ID,"standard_name","height"); call erreur(status,.TRUE.,"put_att_z_ID")
status = NF90_PUT_ATT(fidS,z_ID,"units","m"); call erreur(status,.TRUE.,"put_att_z_ID")
status = NF90_PUT_ATT(fidS,y_ID,"bounds","y_bnds"); call erreur(status,.TRUE.,"put_att_y_ID")
status = NF90_PUT_ATT(fidS,y_ID,"axis","Y"); call erreur(status,.TRUE.,"put_att_y_ID")
status = NF90_PUT_ATT(fidS,y_ID,"long_name","y coordinate of projection"); call erreur(status,.TRUE.,"put_att_y_ID")
status = NF90_PUT_ATT(fidS,y_ID,"standard_name","projection_y_coordinate"); call erreur(status,.TRUE.,"put_att_y_ID")
status = NF90_PUT_ATT(fidS,y_ID,"units","m"); call erreur(status,.TRUE.,"put_att_y_ID")
status = NF90_PUT_ATT(fidS,x_ID,"bounds","x_bnds"); call erreur(status,.TRUE.,"put_att_x_ID")
status = NF90_PUT_ATT(fidS,x_ID,"axis","X"); call erreur(status,.TRUE.,"put_att_x_ID")
status = NF90_PUT_ATT(fidS,x_ID,"long_name","x coordinate of projection"); call erreur(status,.TRUE.,"put_att_x_ID")
status = NF90_PUT_ATT(fidS,x_ID,"standard_name","projection_x_coordinate"); call erreur(status,.TRUE.,"put_att_x_ID")
status = NF90_PUT_ATT(fidS,x_ID,"units","m"); call erreur(status,.TRUE.,"put_att_x_ID")

status = NF90_PUT_ATT(fidS,NF90_GLOBAL,"history","Created using extract_Nico_Kaitlin_warm.f90"); call erreur(status,.TRUE.,"put_att_GLOBAL1_ID")
status = NF90_PUT_ATT(fidS,NF90_GLOBAL,"description",TRIM(desc)); call erreur(status,.TRUE.,"put_att_GLOBAL2_ID")
 
status = NF90_ENDDEF(fidS); call erreur(status,.TRUE.,"fin_definition_S") 
 
status = NF90_PUT_VAR(fidS,sal_ID,sal); call erreur(status,.TRUE.,"var_sal_ID")
status = NF90_PUT_VAR(fidS,sal_std_ID,sal_std); call erreur(status,.TRUE.,"var_sal_std_ID")
status = NF90_PUT_VAR(fidS,x_ID,x); call erreur(status,.TRUE.,"var_x_ID")
status = NF90_PUT_VAR(fidS,y_ID,y); call erreur(status,.TRUE.,"var_y_ID")
status = NF90_PUT_VAR(fidS,z_ID,z); call erreur(status,.TRUE.,"var_z_ID")
status = NF90_PUT_VAR(fidS,x_bnds_ID,x_bnds); call erreur(status,.TRUE.,"var_x_bnds_ID")
status = NF90_PUT_VAR(fidS,y_bnds_ID,y_bnds); call erreur(status,.TRUE.,"var_y_bnds_ID")
status = NF90_PUT_VAR(fidS,z_bnds_ID,z_bnds); call erreur(status,.TRUE.,"var_z_bnds_ID")

status = NF90_CLOSE(fidS); call erreur(status,.TRUE.,"final")

DEALLOCATE( sal, sal_std )

!=======================================================================================================
! TEMPERATURE
!=======================================================================================================

ALLOCATE(  tem(mx,my,mz), tem_std(mx,my,mz)  )
ALLOCATE(  temMITgcm(mx,my,mz)  )
ALLOCATE(  temMITgcm_std(mx,my,mz)  )
ALLOCATE(  tmp(mx,my,mz)  )
ALLOCATE(  temNEMO(mx,my,mz)  )
ALLOCATE(  temNEMO_std(mx,my,mz)  )

!---------------------------------------
! Read temperature climatology
 
write(*,*) 'Reading ', TRIM(file_in_Tclim)
 
status = NF90_OPEN(TRIM(file_in_Tclim),0,fidTclim); call erreur(status,.TRUE.,"read")
 
status = NF90_INQ_VARID(fidTclim,"thetao",tem_ID); call erreur(status,.TRUE.,"inq_thetao_ID")
 
status = NF90_GET_VAR(fidTclim,tem_ID,tem); call erreur(status,.TRUE.,"getvar_thetao")
 
status = NF90_CLOSE(fidTclim); call erreur(status,.TRUE.,"close_file")

!---------------------------------------
! Read MITgcm data:

N_MITgcm = 0
temMITgcm = 0.e0
temMITgcm_std = 0.e0

do year=yeari_MITgcm,yearf_MITgcm

  write(file_in_MITgcm,101) year 

  write(*,*) 'Reading ', TRIM(file_in_MITgcm)
   
  status = NF90_OPEN(TRIM(file_in_MITgcm),0,fidMITgcm); call erreur(status,.TRUE.,"read")
   
  status = NF90_INQ_VARID(fidMITgcm,"temperature",temMITgcm_ID); call erreur(status,.TRUE.,"inq_temMITgcm_ID")
   
  status = NF90_GET_VAR(fidMITgcm,temMITgcm_ID,tmp,start=(/1,1,2/),count=(/mx,my,mz/),stride=(/1,1,3/)); call erreur(status,.TRUE.,"getvar_temMITgcm")
  temMITgcm = temMITgcm + tmp
  temMITgcm_std = temMITgcm_std + tmp**2
   
  status = NF90_CLOSE(fidMITgcm); call erreur(status,.TRUE.,"close_file")

  N_MITgcm = N_MITgcm + 1

enddo

temMITgcm = temMITgcm / N_MITgcm
temMITgcm_std = temMITgcm_std / N_MITgcm
temMITgcm_std = ( temMITgcm_std - temMITgcm**2 )**0.5

!---------------------------------------
! Read Nico's Amundsen

temNEMO = 0.e0
temNEMO_std = 0.e0
N_NEMO = 0

do kk=1,3

  if ( kk .eq. 1 ) then
     exprt='A'
  elseif ( kk .eq. 2 ) then
     exprt='B'
  elseif ( kk .eq. 3 ) then
     exprt='C' 
  endif

  write(file_in_NEMO,102) exprt
 
  write(*,*) 'Reading ', TRIM(file_in_NEMO)
   
  status = NF90_OPEN(TRIM(file_in_NEMO),0,fidNEMO); call erreur(status,.TRUE.,"read")
   
  status = NF90_INQ_VARID(fidNEMO,"thetao",temNEMO_ID); call erreur(status,.TRUE.,"inq_temNEMO_ID")
   
  status = NF90_GET_VAR(fidNEMO,temNEMO_ID,tmp); call erreur(status,.TRUE.,"getvar_temNEMO")
  temNEMO = temNEMO + tmp
  temNEMO_std = temNEMO_std + tmp**2
   
  status = NF90_CLOSE(fidNEMO); call erreur(status,.TRUE.,"close_file")

  N_NEMO = N_NEMO + 1

enddo

temNEMO = temNEMO / N_NEMO
temNEMO_std = temNEMO_std / N_NEMO
temNEMO_std = ( temNEMO_std - temNEMO**2 )**0.5

!---------------------------------------
! Replace climatology with model outputs in some basins

do i=1,mx
do j=1,my

  if ( basinNumber(i,j) .ge. 12 ) then ! Naughten (not taking basin 0 as Beaudouin is missing in the basin)
    tem(i,j,:) = temMITgcm(i,j,:)
    tem_std(i,j,:) = temMITgcm_std(i,j,:)
  elseif ( basinNumber(i,j) .ge. 8 .and. basinNumber(i,j) .le. 10 ) then ! Jourdain
    tem(i,j,:) = temNEMO(i,j,:)
    tem_std(i,j,:) = temNEMO_std(i,j,:)
  endif

  do kk=1,mz
    if ( isnan(tem(i,j,kk)) .or. tem(i,j,kk) .lt. -50.0 .or. tem(i,j,kk) .gt. 50.0 ) then
      tem(i,j,kk) = miss
    endif
    if ( isnan(tem_std(i,j,kk)) .or. tem(i,j,kk) .lt. -50.0 .or. tem(i,j,kk) .gt. 50.0 ) then
      tem_std(i,j,kk) = miss
    endif
  enddo

  ! masking ice shelf cavities and grounded ice
  if ( ice_frac(i,j) .gt. 0.5 ) then
    tem(i,j,:) = miss
    tem_std(i,j,:) = miss
  endif

enddo
enddo 

DEALLOCATE( temMITgcm, temMITgcm_std, tmp, temNEMO, temNEMO_std )
 
!---------------------------------------
! Writing new netcdf file :
 
write(*,*) 'Creating ', TRIM(file_out_T)
 
status = NF90_CREATE(TRIM(file_out_T),NF90_NOCLOBBER,fidT); call erreur(status,.TRUE.,'create')
 
status = NF90_DEF_DIM(fidT,"x",mx,dimID_x); call erreur(status,.TRUE.,"def_dimID_x")
status = NF90_DEF_DIM(fidT,"y",my,dimID_y); call erreur(status,.TRUE.,"def_dimID_y")
status = NF90_DEF_DIM(fidT,"z",mz,dimID_z); call erreur(status,.TRUE.,"def_dimID_z")
status = NF90_DEF_DIM(fidT,"bnds",mbnds,dimID_bnds); call erreur(status,.TRUE.,"def_dimID_bnds")

status = NF90_DEF_VAR(fidT,"thetao",NF90_FLOAT,(/dimID_x,dimID_y,dimID_z/),tem_ID); call erreur(status,.TRUE.,"def_var_tem_ID")
status = NF90_DEF_VAR(fidT,"thetao_uncert",NF90_FLOAT,(/dimID_x,dimID_y,dimID_z/),tem_std_ID); call erreur(status,.TRUE.,"def_var_tem_std_ID")
status = NF90_DEF_VAR(fidT,"x",NF90_DOUBLE,(/dimID_x/),x_ID); call erreur(status,.TRUE.,"def_var_x_ID")
status = NF90_DEF_VAR(fidT,"y",NF90_DOUBLE,(/dimID_y/),y_ID); call erreur(status,.TRUE.,"def_var_y_ID")
status = NF90_DEF_VAR(fidT,"z",NF90_DOUBLE,(/dimID_z/),z_ID); call erreur(status,.TRUE.,"def_var_z_ID")
status = NF90_DEF_VAR(fidT,"x_bnds",NF90_DOUBLE,(/dimID_bnds,dimID_x/),x_bnds_ID); call erreur(status,.TRUE.,"def_var_x_bnds_ID")
status = NF90_DEF_VAR(fidT,"y_bnds",NF90_DOUBLE,(/dimID_bnds,dimID_y/),y_bnds_ID); call erreur(status,.TRUE.,"def_var_y_bnds_ID")
status = NF90_DEF_VAR(fidT,"z_bnds",NF90_DOUBLE,(/dimID_bnds,dimID_z/),z_bnds_ID); call erreur(status,.TRUE.,"def_var_z_bnds_ID")
 
status = NF90_PUT_ATT(fidT,tem_ID,"units","degC"); call erreur(status,.TRUE.,"put_att_tem_ID")
status = NF90_PUT_ATT(fidT,tem_ID,"long_name","Sea Water Potential Temperature"); call erreur(status,.TRUE.,"put_att_tem_ID")
status = NF90_PUT_ATT(fidT,tem_ID,"_FillValue",miss); call erreur(status,.TRUE.,"put_att_tem_ID")
status = NF90_PUT_ATT(fidT,tem_std_ID,"units","degC"); call erreur(status,.TRUE.,"put_att_tem_std_ID")
status = NF90_PUT_ATT(fidT,tem_std_ID,"long_name","Sea Water Potential Temperature stadard deviation"); call erreur(status,.TRUE.,"put_att_tem_std_ID")
status = NF90_PUT_ATT(fidT,tem_std_ID,"comment","Standard deviation over 20 years for MITgcm and over three 21-year means for NEMO"); call erreur(status,.TRUE.,"put_att_tem_std_ID")
status = NF90_PUT_ATT(fidT,tem_std_ID,"_FillValue",miss); call erreur(status,.TRUE.,"put_att_tem_std_ID")
status = NF90_PUT_ATT(fidT,z_ID,"bounds","z_bnds"); call erreur(status,.TRUE.,"put_att_z_ID")
status = NF90_PUT_ATT(fidT,z_ID,"axis","Z"); call erreur(status,.TRUE.,"put_att_z_ID")
status = NF90_PUT_ATT(fidT,z_ID,"positive","up"); call erreur(status,.TRUE.,"put_att_z_ID")
status = NF90_PUT_ATT(fidT,z_ID,"long_name","height relative to sea surface (positive up)"); call erreur(status,.TRUE.,"put_att_z_ID")
status = NF90_PUT_ATT(fidT,z_ID,"standard_name","height"); call erreur(status,.TRUE.,"put_att_z_ID")
status = NF90_PUT_ATT(fidT,z_ID,"units","m"); call erreur(status,.TRUE.,"put_att_z_ID")
status = NF90_PUT_ATT(fidT,y_ID,"bounds","y_bnds"); call erreur(status,.TRUE.,"put_att_y_ID")
status = NF90_PUT_ATT(fidT,y_ID,"axis","Y"); call erreur(status,.TRUE.,"put_att_y_ID")
status = NF90_PUT_ATT(fidT,y_ID,"long_name","y coordinate of projection"); call erreur(status,.TRUE.,"put_att_y_ID")
status = NF90_PUT_ATT(fidT,y_ID,"standard_name","projection_y_coordinate"); call erreur(status,.TRUE.,"put_att_y_ID")
status = NF90_PUT_ATT(fidT,y_ID,"units","m"); call erreur(status,.TRUE.,"put_att_y_ID")
status = NF90_PUT_ATT(fidT,x_ID,"bounds","x_bnds"); call erreur(status,.TRUE.,"put_att_x_ID")
status = NF90_PUT_ATT(fidT,x_ID,"axis","X"); call erreur(status,.TRUE.,"put_att_x_ID")
status = NF90_PUT_ATT(fidT,x_ID,"long_name","x coordinate of projection"); call erreur(status,.TRUE.,"put_att_x_ID")
status = NF90_PUT_ATT(fidT,x_ID,"standard_name","projection_x_coordinate"); call erreur(status,.TRUE.,"put_att_x_ID")
status = NF90_PUT_ATT(fidT,x_ID,"units","m"); call erreur(status,.TRUE.,"put_att_x_ID")

status = NF90_PUT_ATT(fidT,NF90_GLOBAL,"history","Created using extract_Nico_Kaitlin_warm.f90"); call erreur(status,.TRUE.,"put_att_GLOBAL1_ID")
status = NF90_PUT_ATT(fidT,NF90_GLOBAL,"description",TRIM(desc)); call erreur(status,.TRUE.,"put_att_GLOBAL2_ID")
 
status = NF90_ENDDEF(fidT); call erreur(status,.TRUE.,"fin_definition_T") 
 
status = NF90_PUT_VAR(fidT,tem_ID,tem); call erreur(status,.TRUE.,"var_tem_ID")
status = NF90_PUT_VAR(fidT,tem_std_ID,tem_std); call erreur(status,.TRUE.,"var_tem_std_ID")
status = NF90_PUT_VAR(fidT,x_ID,x); call erreur(status,.TRUE.,"var_x_ID")
status = NF90_PUT_VAR(fidT,y_ID,y); call erreur(status,.TRUE.,"var_y_ID")
status = NF90_PUT_VAR(fidT,z_ID,z); call erreur(status,.TRUE.,"var_z_ID")
status = NF90_PUT_VAR(fidT,x_bnds_ID,x_bnds); call erreur(status,.TRUE.,"var_x_bnds_ID")
status = NF90_PUT_VAR(fidT,y_bnds_ID,y_bnds); call erreur(status,.TRUE.,"var_y_bnds_ID")
status = NF90_PUT_VAR(fidT,z_bnds_ID,z_bnds); call erreur(status,.TRUE.,"var_z_bnds_ID")

status = NF90_CLOSE(fidT); call erreur(status,.TRUE.,"final")

DEALLOCATE( tem, tem_std )
 
!=======================================================================================================
! ICE SHELF BASAL MELT
!=======================================================================================================

ALLOCATE(  mlt(mx,my), mlt_std(mx,my)  )
ALLOCATE(  mltMITgcm(mx,my)  )
ALLOCATE(  mltMITgcm_std(mx,my)  )
ALLOCATE(  txp(mx,my), typ(mx,my), tzp(mx,my)  )
ALLOCATE(  mltNEMO(mx,my), mltNEMO_std(mx,my)  )
ALLOCATE(  sftflf(mx,my)  )

!---------------------------------------
! Read MITgcm data:

N_MITgcm = 0
mltMITgcm = 0.e0
mltMITgcm_std = 0.e0

do year=yeari_MITgcm,yearf_MITgcm

  write(file_in_MITgcm,101) year 

  write(*,*) 'Reading ', TRIM(file_in_MITgcm)
   
  status = NF90_OPEN(TRIM(file_in_MITgcm),0,fidMITgcm); call erreur(status,.TRUE.,"read")
   
  status = NF90_INQ_VARID(fidMITgcm,"basal_melt",mltMITgcm_ID); call erreur(status,.TRUE.,"inq_mltMITgcm_ID")
   
  status = NF90_GET_VAR(fidMITgcm,mltMITgcm_ID,txp); call erreur(status,.TRUE.,"getvar_mltMITgcm")
  mltMITgcm = mltMITgcm + (txp*920.)       ! m/yr -> kg/m2/yr
  mltMITgcm_std = mltMITgcm_std + (txp*920.)**2
   
  status = NF90_CLOSE(fidMITgcm); call erreur(status,.TRUE.,"close_file")

  N_MITgcm = N_MITgcm + 1

enddo

mltMITgcm = mltMITgcm / N_MITgcm
mltMITgcm_std = mltMITgcm_std / N_MITgcm
mltMITgcm_std = ( mltMITgcm_std - mltMITgcm**2 )**0.5

!---------------------------------------
! Read Nico's Amundsen

mltNEMO = 0.e0
mltNEMO_std = 0.e0
N_NEMO = 0

do kk=1,3

  if ( kk .eq. 1 ) then
     exprt='A'
  elseif ( kk .eq. 2 ) then
     exprt='B'
  elseif ( kk .eq. 3 ) then
     exprt='C' 
  endif

  write(file_in_NEMO,102) exprt
 
  write(*,*) 'Reading ', TRIM(file_in_NEMO)
   
  status = NF90_OPEN(TRIM(file_in_NEMO),0,fidNEMO); call erreur(status,.TRUE.,"read")
   
  status = NF90_INQ_VARID(fidNEMO,"libmassbffl",mltNEMO_ID); call erreur(status,.TRUE.,"inq_mltNEMO_ID")
  status = NF90_INQ_VARID(fidNEMO,"sftflf",sftflf_ID); call erreur(status,.TRUE.,"inq_sftflf_ID")
   
  status = NF90_GET_VAR(fidNEMO,mltNEMO_ID,txp); call erreur(status,.TRUE.,"getvar_mltNEMO")
  status = NF90_GET_VAR(fidNEMO,sftflf_ID,typ); call erreur(status,.TRUE.,"getvar_sftflf")
  tzp = txp * typ / 100. * 86400. * 365.25 ! assuming output melt rate has to be provided per area of ocean and in kg/m2/a
  mltNEMO = mltNEMO + tzp
  mltNEMO_std = mltNEMO_std + tzp**2
   
  status = NF90_CLOSE(fidNEMO); call erreur(status,.TRUE.,"close_file")

  N_NEMO = N_NEMO + 1

enddo

mltNEMO = mltNEMO / N_NEMO
mltNEMO_std = mltNEMO_std / N_NEMO
mltNEMO_std = ( mltNEMO_std - mltNEMO**2 )**0.5

!---------------------------------------
! Merging different basins

do i=1,mx
do j=1,my

  if ( basinNumber(i,j) .ge. 12 ) then ! Naughten (not taking basin 0 as Beaudouin is missing in the basin)
    mlt(i,j) = mltMITgcm(i,j)
    mlt_std(i,j) = mltMITgcm_std(i,j)
  elseif ( basinNumber(i,j) .ge. 8 .and. basinNumber(i,j) .le. 10 ) then ! Jourdain
    mlt(i,j) = mltNEMO(i,j)
    mlt_std(i,j) = mltNEMO_std(i,j)
  else ! climatology
    mlt(i,j) = 0.e0
    mlt_std(i,j) = 0.e0
  endif

  if ( isnan(mlt(i,j)) .or. abs(mlt(i,j)) .gt. 1.e7 ) then
    mlt(i,j) = 0.e0
  endif
  if ( isnan(mlt_std(i,j)) .or. abs(mlt(i,j)) .gt. 1.e7 ) then
    mlt_std(i,j) = 0.e0
  endif

enddo
enddo 

DEALLOCATE(txp, typ, tzp, mltMITgcm, mltMITgcm_std, mltNEMO, mltNEMO_std, sftflf )
 
!---------------------------------------
! Writing new netcdf file :
 
write(*,*) 'Creating ', TRIM(file_out_m)
 
status = NF90_CREATE(TRIM(file_out_m),NF90_NOCLOBBER,fidm); call erreur(status,.TRUE.,'create')
 
status = NF90_DEF_DIM(fidm,"x",mx,dimID_x); call erreur(status,.TRUE.,"def_dimID_x")
status = NF90_DEF_DIM(fidm,"y",my,dimID_y); call erreur(status,.TRUE.,"def_dimID_y")
status = NF90_DEF_DIM(fidm,"bnds",mbnds,dimID_bnds); call erreur(status,.TRUE.,"def_dimID_bnds")
  
status = NF90_DEF_VAR(fidm,"basal_melt",NF90_FLOAT,(/dimID_x,dimID_y/),mlt_ID); call erreur(status,.TRUE.,"def_var_mlt_ID")
status = NF90_DEF_VAR(fidm,"basal_melt_uncert",NF90_FLOAT,(/dimID_x,dimID_y/),mlt_std_ID); call erreur(status,.TRUE.,"def_var_mlt_std_ID")
status = NF90_DEF_VAR(fidm,"x",NF90_DOUBLE,(/dimID_x/),x_ID); call erreur(status,.TRUE.,"def_var_x_ID")
status = NF90_DEF_VAR(fidm,"y",NF90_DOUBLE,(/dimID_y/),y_ID); call erreur(status,.TRUE.,"def_var_y_ID")
status = NF90_DEF_VAR(fidm,"x_bnds",NF90_DOUBLE,(/dimID_bnds,dimID_x/),x_bnds_ID); call erreur(status,.TRUE.,"def_var_x_bnds_ID")
status = NF90_DEF_VAR(fidm,"y_bnds",NF90_DOUBLE,(/dimID_bnds,dimID_y/),y_bnds_ID); call erreur(status,.TRUE.,"def_var_y_bnds_ID")
 
status = NF90_PUT_ATT(fidm,mlt_ID,"units","kg/m2/a"); call erreur(status,.TRUE.,"put_att_mlt_ID")
status = NF90_PUT_ATT(fidm,mlt_ID,"long_name","Ice shelf melt rate"); call erreur(status,.TRUE.,"put_att_mlt_ID")
status = NF90_PUT_ATT(fidm,mlt_ID,"comment","positive means melting"); call erreur(status,.TRUE.,"put_att_mlt_ID")
status = NF90_PUT_ATT(fidm,mlt_ID,"_FillValue",miss); call erreur(status,.TRUE.,"put_att_mlt_ID")
status = NF90_PUT_ATT(fidm,mlt_std_ID,"units","kg/m2/a"); call erreur(status,.TRUE.,"put_att_mlt_std_ID")
status = NF90_PUT_ATT(fidm,mlt_std_ID,"long_name","Ice shelf melt rate standard deviation"); call erreur(status,.TRUE.,"put_att_mlt_std_ID")
status = NF90_PUT_ATT(fidm,mlt_std_ID,"comment","Standard deviation over 20 years for MITgcm and over three 21-year means for NEMO"); call erreur(status,.TRUE.,"put_att_mlt_std_ID")
status = NF90_PUT_ATT(fidm,mlt_std_ID,"_FillValue",miss); call erreur(status,.TRUE.,"put_att_mlt_std_ID")
status = NF90_PUT_ATT(fidm,y_ID,"bounds","y_bnds"); call erreur(status,.TRUE.,"put_att_y_ID")
status = NF90_PUT_ATT(fidm,y_ID,"axis","Y"); call erreur(status,.TRUE.,"put_att_y_ID")
status = NF90_PUT_ATT(fidm,y_ID,"long_name","y coordinate of projection"); call erreur(status,.TRUE.,"put_att_y_ID")
status = NF90_PUT_ATT(fidm,y_ID,"standard_name","projection_y_coordinate"); call erreur(status,.TRUE.,"put_att_y_ID")
status = NF90_PUT_ATT(fidm,y_ID,"units","m"); call erreur(status,.TRUE.,"put_att_y_ID")
status = NF90_PUT_ATT(fidm,x_ID,"bounds","x_bnds"); call erreur(status,.TRUE.,"put_att_x_ID")
status = NF90_PUT_ATT(fidm,x_ID,"axis","X"); call erreur(status,.TRUE.,"put_att_x_ID")
status = NF90_PUT_ATT(fidm,x_ID,"long_name","x coordinate of projection"); call erreur(status,.TRUE.,"put_att_x_ID")
status = NF90_PUT_ATT(fidm,x_ID,"standard_name","projection_x_coordinate"); call erreur(status,.TRUE.,"put_att_x_ID")
status = NF90_PUT_ATT(fidm,x_ID,"units","m"); call erreur(status,.TRUE.,"put_att_x_ID")

status = NF90_PUT_ATT(fidm,NF90_GLOBAL,"history","Created using extract_Nico_Kaitlin_warm.f90"); call erreur(status,.TRUE.,"put_att_GLOBAL1_ID")
status = NF90_PUT_ATT(fidm,NF90_GLOBAL,"description",TRIM(desc)); call erreur(status,.TRUE.,"put_att_GLOBAL2_ID")
 
status = NF90_ENDDEF(fidm); call erreur(status,.TRUE.,"fin_definition_m") 
 
status = NF90_PUT_VAR(fidm,mlt_ID,mlt); call erreur(status,.TRUE.,"var_mlt_ID")
status = NF90_PUT_VAR(fidm,mlt_std_ID,mlt_std); call erreur(status,.TRUE.,"var_mlt_std_ID")
status = NF90_PUT_VAR(fidm,x_ID,x); call erreur(status,.TRUE.,"var_x_ID")
status = NF90_PUT_VAR(fidm,y_ID,y); call erreur(status,.TRUE.,"var_y_ID")
status = NF90_PUT_VAR(fidm,x_bnds_ID,x_bnds); call erreur(status,.TRUE.,"var_x_bnds_ID")
status = NF90_PUT_VAR(fidm,y_bnds_ID,y_bnds); call erreur(status,.TRUE.,"var_y_bnds_ID")

status = NF90_CLOSE(fidm); call erreur(status,.TRUE.,"final")

DEALLOCATE( mlt, mlt_std ) 

end program modif

!================================================

SUBROUTINE erreur(iret, lstop, chaine)
  ! pour les messages d'erreur
  USE netcdf
  INTEGER, INTENT(in)                     :: iret
  LOGICAL, INTENT(in)                     :: lstop
  CHARACTER(LEN=*), INTENT(in)            :: chaine
  !
  CHARACTER(LEN=80)                       :: message
  !
  IF ( iret .NE. 0 ) THEN
    WRITE(*,*) 'ROUTINE: ', TRIM(chaine)
    WRITE(*,*) 'ERREUR: ', iret
    message=NF90_STRERROR(iret)
    WRITE(*,*) 'CA VEUT DIRE:',TRIM(message)
    IF ( lstop ) STOP
  ENDIF
  !
END SUBROUTINE erreur
