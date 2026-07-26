program modif
 
USE netcdf
 
IMPLICIT NONE
 
INTEGER :: fidMITwed, status, dimID_x, dimID_y, dimID_z, dimID_time, mx, my, mz, mtime, mltMITwed_ID,   &
&          salMITwed_ID, temMITwed_ID, x_ID, y_ID, z_ID, time_ID, fidm, basinNumber_ID, fidB, N_MITwed, &
&          fidMITamu, salMITamu_ID, temMITamu_ID, mltMITamu_ID, kk, N_MITamu, i, j, fidS, fidT,         &
&          fidSclim, fidTclim, dimID_bnds, mbnds, z_bnds_ID, y_bnds_ID, x_bnds_ID, sal_ID, tem_ID, year,&
&          mlt_ID, mlt_std_ID, yeari_MITwed, yearf_MITwed, fidBM, rock_frac_ID, ice_frac_ID, sal_std_ID,&
&          tem_std_ID, yeari_MITamu, yearf_MITamu, ens
 
CHARACTER(LEN=1) :: exprt

CHARACTER(LEN=180) :: file_in_MITwed, file_in_MITamu, file_in_B, file_out_S, file_out_T, file_out_m, &
&                     file_in_Sclim, file_in_Tclim, file_in_BM

CHARACTER(LEN=300) :: desc

INTEGER*4,ALLOCATABLE,DIMENSION(:,:) :: basinNumber
 
REAL*8,ALLOCATABLE,DIMENSION(:) :: x, y, z

REAL*8,ALLOCATABLE,DIMENSION(:,:) :: z_bnds, y_bnds, x_bnds
 
REAL*4,ALLOCATABLE,DIMENSION(:,:) :: mltMITwed, txp, mltMITamu, mltMITwed_std, mltMITamu_std,    &
&                                    mlt, mlt_std, rock_frac, ice_frac
 
REAL*4,ALLOCATABLE,DIMENSION(:,:,:) :: salMITwed, temMITwed, salMITwed_std, temMITwed_std, tmp, sal, tem, &
&                                      salMITamu, temMITamu, salMITamu_std, temMITamu_std, sal_std, tem_std

REAL*4 :: miss

!---------------------------------------

file_out_S = 'Naughten_MITamu-MITwed_cold_S.nc'
file_out_T = 'Naughten_MITamu-MITwed_cold_T.nc'
file_out_m = 'Naughten_MITamu-MITwed_cold_m.nc'

file_in_B  = '/scratchu/njourdain/ISMIP7_REGIONAL_OCEAN_DATA/DATA/basin_numbers_ismip8km_v2.nc'
file_in_BM  = '/scratchu/njourdain/ISMIP7_REGIONAL_OCEAN_DATA/DATA/BedMachineAntarctica-v3_AIS_obs_ocean_topography_v3.nc'
file_in_Sclim  = '/scratchu/njourdain/ISMIP7_REGIONAL_OCEAN_DATA/DATA/so_AIS_obs_ocean_climatology_zhou_annual_06_nov_v4_1972-2024.nc'
file_in_Tclim  = '/scratchu/njourdain/ISMIP7_REGIONAL_OCEAN_DATA/DATA/thetao_AIS_obs_ocean_climatology_zhou_annual_06_nov_v4_1972-2024.nc'

101 FORMAT('/scratchu/njourdain/ISMIP7_REGIONAL_OCEAN_DATA/DATA/Kaitlin/WED/MITgcm_WS_abrupt-4xCO2_',i4,'.nc')
102 FORMAT('/scratchu/njourdain/ISMIP7_REGIONAL_OCEAN_DATA/DATA/Kaitlin/AMU_RCP85/MITgcm_ASE_RCP85_ens',i2.2,'_',i4,'.nc')

yeari_MITwed = 1850
yearf_MITwed = 1869
yeari_MITamu = 2006
yearf_MITamu = 2007

mz = 30

111 FORMAT("Merge of MITgcm-Weddell-abrupt4xCO2 over ",i4,"-",i4," (Naughten 2021) in basins 0, 12, 13, 14, 15 and of MITgcm-Amundsen-rcp85 over 10 members of ",i4,"-",i4," (Naughten 2023) in basins 8, 9, 10; climatology elsewhere")
write(desc,111) yeari_MITwed, yearf_MITwed, yeari_MITamu, yearf_MITamu
 
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
ALLOCATE(  salMITwed(mx,my,mz)  )
ALLOCATE(  salMITwed_std(mx,my,mz)  )
ALLOCATE(  salMITamu(mx,my,mz)  )
ALLOCATE(  salMITamu_std(mx,my,mz)  )
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
! Read MITwed data:

N_MITwed = 0
salMITwed = 0.e0
salMITwed_std = 0.e0

do year=yeari_MITwed,yearf_MITwed

  write(file_in_MITwed,101) year 

  write(*,*) 'Reading ', TRIM(file_in_MITwed)
   
  status = NF90_OPEN(TRIM(file_in_MITwed),0,fidMITwed); call erreur(status,.TRUE.,"read")
   
  status = NF90_INQ_VARID(fidMITwed,"salinity",salMITwed_ID); call erreur(status,.TRUE.,"inq_salMITwed_ID")
  status = NF90_INQ_VARID(fidMITwed,"z",z_ID); call erreur(status,.TRUE.,"inq_z_ID")
   
  status = NF90_GET_VAR(fidMITwed,salMITwed_ID,tmp,start=(/1,1,2/),count=(/mx,my,mz/),stride=(/1,1,3/)); call erreur(status,.TRUE.,"getvar_salMITwed")
  salMITwed = salMITwed + tmp
  salMITwed_std = salMITwed_std + tmp**2
  status = NF90_GET_VAR(fidMITwed,z_ID,z,start=(/2/),count=(/mz/),stride=(/3/)); call erreur(status,.TRUE.,"getvar_z")
   
  status = NF90_CLOSE(fidMITwed); call erreur(status,.TRUE.,"close_file")

  N_MITwed = N_MITwed + 1

enddo

salMITwed = salMITwed / N_MITwed
salMITwed_std = salMITwed_std / N_MITwed
salMITwed_std = ( salMITwed_std - salMITwed**2 )**0.5

!---------------------------------------
! Read MITamu data

salMITamu = 0.e0
salMITamu_std = 0.e0
N_MITamu = 0

do year = yeari_MITamu, yearf_MITamu
do ens = 1,10

  write(file_in_MITamu,102) ens, year
 
  write(*,*) 'Reading ', TRIM(file_in_MITamu)
   
  status = NF90_OPEN(TRIM(file_in_MITamu),0,fidMITamu); call erreur(status,.TRUE.,"read")
   
  status = NF90_INQ_VARID(fidMITamu,"salinity",salMITamu_ID); call erreur(status,.TRUE.,"inq_salMITamu_ID")
   
  status = NF90_GET_VAR(fidMITamu,salMITamu_ID,tmp); call erreur(status,.TRUE.,"getvar_salMITamu")
  salMITamu = salMITamu + tmp
  salMITamu_std = salMITamu_std + tmp**2
   
  status = NF90_CLOSE(fidMITamu); call erreur(status,.TRUE.,"close_file")

  N_MITamu = N_MITamu + 1

enddo
enddo

salMITamu = salMITamu / N_MITamu
salMITamu_std = salMITamu_std / N_MITamu
salMITamu_std = ( salMITamu_std - salMITamu**2 )**0.5

!---------------------------------------
! Replace climatology with model outputs in some basins

do i=1,mx
do j=1,my

  if ( basinNumber(i,j) .ge. 12 ) then ! Naughten (not taking basin 0 as Beaudouin is missing in the basin)
    sal(i,j,:) = salMITwed(i,j,:)
    sal_std(i,j,:) = salMITwed_std(i,j,:)
  elseif ( basinNumber(i,j) .ge. 8 .and. basinNumber(i,j) .le. 10 ) then ! Jourdain
    sal(i,j,:) = salMITamu(i,j,:)
    sal_std(i,j,:) = salMITamu_std(i,j,:)
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

DEALLOCATE( salMITwed, salMITwed_std, salMITamu, salMITamu_std, tmp )

!---------------------------------------
! Writing new netcdf file :
 
write(*,*) 'Creating ', TRIM(file_out_S)
 
status = NF90_CREATE(TRIM(file_out_S),NF90_NOCLOBBER,fidS); call erreur(status,.TRUE.,'create')
 
status = NF90_DEF_DIM(fidS,"x",mx,dimID_x); call erreur(status,.TRUE.,"def_dimID_x")
status = NF90_DEF_DIM(fidS,"y",my,dimID_y); call erreur(status,.TRUE.,"def_dimID_y")
status = NF90_DEF_DIM(fidS,"z",mz,dimID_z); call erreur(status,.TRUE.,"def_dimID_z")
status = NF90_DEF_DIM(fidS,"bnds",mbnds,dimID_bnds); call erreur(status,.TRUE.,"def_dimID_bnds")
  
status = NF90_DEF_VAR(fidS,"so",NF90_FLOAT,(/dimID_x,dimID_y,dimID_z/),sal_ID); call erreur(status,.TRUE.,"def_var_salMITwed_ID")
status = NF90_DEF_VAR(fidS,"so_uncert",NF90_FLOAT,(/dimID_x,dimID_y,dimID_z/),sal_std_ID); call erreur(status,.TRUE.,"def_var_salMITwed_ID")
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
status = NF90_PUT_ATT(fidS,sal_std_ID,"comment","Standard deviation over 20 years for MITwed and over ten times 2 years for MITamu"); call erreur(status,.TRUE.,"put_att_sal_std_ID")
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

status = NF90_PUT_ATT(fidS,NF90_GLOBAL,"history","Created using extract_Kaitlin_2_cold.f90"); call erreur(status,.TRUE.,"put_att_GLOBAL1_ID")
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
ALLOCATE(  temMITwed(mx,my,mz)  )
ALLOCATE(  temMITwed_std(mx,my,mz)  )
ALLOCATE(  tmp(mx,my,mz)  )
ALLOCATE(  temMITamu(mx,my,mz)  )
ALLOCATE(  temMITamu_std(mx,my,mz)  )

!---------------------------------------
! Read temperature climatology
 
write(*,*) 'Reading ', TRIM(file_in_Tclim)
 
status = NF90_OPEN(TRIM(file_in_Tclim),0,fidTclim); call erreur(status,.TRUE.,"read")
 
status = NF90_INQ_VARID(fidTclim,"thetao",tem_ID); call erreur(status,.TRUE.,"inq_thetao_ID")
 
status = NF90_GET_VAR(fidTclim,tem_ID,tem); call erreur(status,.TRUE.,"getvar_thetao")
 
status = NF90_CLOSE(fidTclim); call erreur(status,.TRUE.,"close_file")

!---------------------------------------
! Read MITwed data:

N_MITwed = 0
temMITwed = 0.e0
temMITwed_std = 0.e0

do year=yeari_MITwed,yearf_MITwed

  write(file_in_MITwed,101) year 

  write(*,*) 'Reading ', TRIM(file_in_MITwed)
   
  status = NF90_OPEN(TRIM(file_in_MITwed),0,fidMITwed); call erreur(status,.TRUE.,"read")
   
  status = NF90_INQ_VARID(fidMITwed,"temperature",temMITwed_ID); call erreur(status,.TRUE.,"inq_temMITwed_ID")
   
  status = NF90_GET_VAR(fidMITwed,temMITwed_ID,tmp,start=(/1,1,2/),count=(/mx,my,mz/),stride=(/1,1,3/)); call erreur(status,.TRUE.,"getvar_temMITwed")
  temMITwed = temMITwed + tmp
  temMITwed_std = temMITwed_std + tmp**2
   
  status = NF90_CLOSE(fidMITwed); call erreur(status,.TRUE.,"close_file")

  N_MITwed = N_MITwed + 1

enddo

temMITwed = temMITwed / N_MITwed
temMITwed_std = temMITwed_std / N_MITwed
temMITwed_std = ( temMITwed_std - temMITwed**2 )**0.5

!---------------------------------------
! Read MITamu data

temMITamu = 0.e0
temMITamu_std = 0.e0
N_MITamu = 0

do year = yeari_MITamu, yearf_MITamu
do ens = 1,10

  write(file_in_MITamu,102) ens, year
 
  write(*,*) 'Reading ', TRIM(file_in_MITamu)
   
  status = NF90_OPEN(TRIM(file_in_MITamu),0,fidMITamu); call erreur(status,.TRUE.,"read")
   
  status = NF90_INQ_VARID(fidMITamu,"temperature",temMITamu_ID); call erreur(status,.TRUE.,"inq_temMITamu_ID")
   
  status = NF90_GET_VAR(fidMITamu,temMITamu_ID,tmp); call erreur(status,.TRUE.,"getvar_temMITamu")
  temMITamu = temMITamu + tmp
  temMITamu_std = temMITamu_std + tmp**2
   
  status = NF90_CLOSE(fidMITamu); call erreur(status,.TRUE.,"close_file")

  N_MITamu = N_MITamu + 1

enddo
enddo

temMITamu = temMITamu / N_MITamu
temMITamu_std = temMITamu_std / N_MITamu
temMITamu_std = ( temMITamu_std - temMITamu**2 )**0.5

!---------------------------------------
! Replace climatology with model outputs in some basins

do i=1,mx
do j=1,my

  if ( basinNumber(i,j) .ge. 12 ) then ! Naughten (not taking basin 0 as Beaudouin is missing in the basin)
    tem(i,j,:) = temMITwed(i,j,:)
    tem_std(i,j,:) = temMITwed_std(i,j,:)
  elseif ( basinNumber(i,j) .ge. 8 .and. basinNumber(i,j) .le. 10 ) then ! Jourdain
    tem(i,j,:) = temMITamu(i,j,:)
    tem_std(i,j,:) = temMITamu_std(i,j,:)
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

DEALLOCATE( temMITwed, temMITwed_std, tmp, temMITamu, temMITamu_std )
 
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
status = NF90_PUT_ATT(fidT,tem_std_ID,"comment","Standard deviation over 20 years for MITwed and over ten times 2 years for MITamu"); call erreur(status,.TRUE.,"put_att_tem_std_ID")
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

status = NF90_PUT_ATT(fidT,NF90_GLOBAL,"history","Created using extract_Kaitlin_2_cold.f90"); call erreur(status,.TRUE.,"put_att_GLOBAL1_ID")
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
ALLOCATE(  mltMITwed(mx,my)  )
ALLOCATE(  mltMITwed_std(mx,my)  )
ALLOCATE(  txp(mx,my) )
ALLOCATE(  mltMITamu(mx,my), mltMITamu_std(mx,my)  )

!---------------------------------------
! Read MITwed data:

N_MITwed = 0
mltMITwed = 0.e0
mltMITwed_std = 0.e0

do year=yeari_MITwed,yearf_MITwed

  write(file_in_MITwed,101) year 

  write(*,*) 'Reading ', TRIM(file_in_MITwed)
   
  status = NF90_OPEN(TRIM(file_in_MITwed),0,fidMITwed); call erreur(status,.TRUE.,"read")
   
  status = NF90_INQ_VARID(fidMITwed,"basal_melt",mltMITwed_ID); call erreur(status,.TRUE.,"inq_mltMITwed_ID")
   
  status = NF90_GET_VAR(fidMITwed,mltMITwed_ID,txp); call erreur(status,.TRUE.,"getvar_mltMITwed")
  mltMITwed = mltMITwed + (txp*920.)       ! m/yr -> kg/m2/yr
  mltMITwed_std = mltMITwed_std + (txp*920.)**2
   
  status = NF90_CLOSE(fidMITwed); call erreur(status,.TRUE.,"close_file")

  N_MITwed = N_MITwed + 1

enddo

mltMITwed = mltMITwed / N_MITwed
mltMITwed_std = mltMITwed_std / N_MITwed
mltMITwed_std = ( mltMITwed_std - mltMITwed**2 )**0.5

!---------------------------------------
! Read MITamu data

mltMITamu = 0.e0
mltMITamu_std = 0.e0
N_MITamu = 0

do year = yeari_MITamu, yearf_MITamu
do ens = 1,10

  write(file_in_MITamu,102) ens, year
 
  write(*,*) 'Reading ', TRIM(file_in_MITamu)
   
  status = NF90_OPEN(TRIM(file_in_MITamu),0,fidMITamu); call erreur(status,.TRUE.,"read")
   
  status = NF90_INQ_VARID(fidMITamu,"basal_melt",mltMITamu_ID); call erreur(status,.TRUE.,"inq_mltMITamu_ID")
   
  status = NF90_GET_VAR(fidMITamu,mltMITamu_ID,txp); call erreur(status,.TRUE.,"getvar_mltMITamu")
  mltMITamu = mltMITamu + txp * 920.  ! m/yr -> kg/m2/yr
  mltMITamu_std = mltMITamu_std + (txp*920.)**2
   
  status = NF90_CLOSE(fidMITamu); call erreur(status,.TRUE.,"close_file")

  N_MITamu = N_MITamu + 1

enddo
enddo

mltMITamu = mltMITamu / N_MITamu
mltMITamu_std = mltMITamu_std / N_MITamu
mltMITamu_std = ( mltMITamu_std - mltMITamu**2 )**0.5

!---------------------------------------
! Merging different basins

do i=1,mx
do j=1,my

  if ( basinNumber(i,j) .ge. 12 ) then ! Naughten (not taking basin 0 as Beaudouin is missing in the basin)
    mlt(i,j) = mltMITwed(i,j)
    mlt_std(i,j) = mltMITwed_std(i,j)
  elseif ( basinNumber(i,j) .ge. 8 .and. basinNumber(i,j) .le. 10 ) then ! Jourdain
    mlt(i,j) = mltMITamu(i,j)
    mlt_std(i,j) = mltMITamu_std(i,j)
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

DEALLOCATE(txp, mltMITwed, mltMITwed_std, mltMITamu, mltMITamu_std )
 
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
status = NF90_PUT_ATT(fidm,mlt_std_ID,"comment","Standard deviation over 20 years for MITwed and over ten times 2 years for MITamu"); call erreur(status,.TRUE.,"put_att_mlt_std_ID")
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

status = NF90_PUT_ATT(fidm,NF90_GLOBAL,"history","Created using extract_Kaitlin_2_cold.f90"); call erreur(status,.TRUE.,"put_att_GLOBAL1_ID")
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
