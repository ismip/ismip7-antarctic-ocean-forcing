program modif

USE netcdf

IMPLICIT NONE

INTEGER :: fidA, status, dimID_x, dimID_y, dimID_z, dimID_time, mx, my, mz, mtime, &
&          time_ID, z_ID, y_ID, x_ID, varin_ID, fidM, fidB, fidT, basinNumber_ID,  &
&          mask_ID, bed_ID, kiter, ki, kj, kz, kt, kbasin, Nbasin, mask_ocean_ID,  &
&          varout_ID, kim1, kip1, kjm1, kjp1, kim2, kip2, kjm2, kjp2, kim3, kip3,  &
&          kjm3, kjp3, Niter2

CHARACTER(LEN=15) :: varnam

CHARACTER(LEN=2048) :: file_in, file_out, file_basin, file_topo, cal, uni, his
CHARACTER(LEN=2048) :: namelist_file

INTEGER :: ios, narg

NAMELIST /horizontal_extrapolation/ file_in, file_out, file_basin, file_topo, varnam

REAL*4 :: miss

REAL*4 :: aa, bb, bbim1jjj, bbip1jjj, bbiiijm1, bbiiijp1, bbim2jjj, bbip2jjj, bbiiijm2, bbiiijp2, &
&                 bbim1jm1, bbim1jp1, bbip1jm1, bbip1jp1, bbim2jp1, bbim2jm1, bbip2jp1, bbip2jm1, &
&                 bbim1jp2, bbip1jp2, bbim1jm2, bbip1jm2, bbim2jm2, bbim2jp2, bbip2jm2, bbip2jp2, &
&                 bbim3jjj, bbip3jjj, bbiiijm3, bbiiijp3

REAL*4,ALLOCATABLE,DIMENSION(:) :: z, y, x

REAL*8,ALLOCATABLE,DIMENSION(:) :: time

INTEGER*8,ALLOCATABLE,DIMENSION(:,:) :: basinNumber

REAL*4,ALLOCATABLE,DIMENSION(:,:) :: mask, bed, mask_ocean

REAL*4,ALLOCATABLE,DIMENSION(:,:,:,:) :: var, var_new

INTEGER,ALLOCATABLE,DIMENSION(:,:) :: mskba

INTEGER,ALLOCATABLE,DIMENSION(:) :: Niter

!----------------------------------------------------------------
! Namelist-driven configuration
! Provide clearly invalid defaults so the user must supply a namelist.
file_in     = '__REQUIRED__'
file_out    = '__REQUIRED__'
file_basin  = '__REQUIRED__'
file_topo   = '__REQUIRED__'
varnam      = '__REQUIRED__'

! Determine namelist file name: first command line argument if present,
! otherwise fallback to a conventional filename.
narg = COMMAND_ARGUMENT_COUNT()
if ( narg >= 1 ) then
  call GET_COMMAND_ARGUMENT(1, namelist_file, status=ios)
  if ( ios /= 0 ) then
    write(*,*) 'ERROR retrieving command argument for namelist file.'
    stop 1
  end if
  if ( LEN_TRIM(namelist_file) == 0 ) namelist_file = 'extrapolate_horizontal.nml'
else
  namelist_file = 'extrapolate_horizontal.nml'
end if

open(unit=21, file=TRIM(namelist_file), status='old', action='read', iostat=ios)
if ( ios /= 0 ) then
  write(*,*) 'ERROR: cannot open namelist file: ', TRIM(namelist_file)
  stop 1
end if

read(21, nml=horizontal_extrapolation, iostat=ios)
if ( ios /= 0 ) then
  write(*,*) 'ERROR: reading namelist "horizontal_extrapolation" from file: ', TRIM(namelist_file)
  stop 1
end if
close(21)

if ( TRIM(file_in)    == '__REQUIRED__' .or. &
    TRIM(file_out)   == '__REQUIRED__' .or. &
    TRIM(file_basin) == '__REQUIRED__' .or. &
    TRIM(file_topo)  == '__REQUIRED__' .or. &
    TRIM(varnam)     == '__REQUIRED__' ) then
  write(*,*) 'ERROR: One or more required namelist entries were not provided.'
  write(*,*) '       file_in     = ', TRIM(file_in)
  write(*,*) '       file_out    = ', TRIM(file_out)
  write(*,*) '       file_basin  = ', TRIM(file_basin)
  write(*,*) '       file_topo   = ', TRIM(file_topo)
  write(*,*) '       varnam      = ', TRIM(varnam)
  stop 1
end if

write(*,*) 'Using namelist: ', TRIM(namelist_file)
write(*,*) '  file_in    = ', TRIM(file_in)
write(*,*) '  file_out   = ', TRIM(file_out)
write(*,*) '  file_basin = ', TRIM(file_basin)
write(*,*) '  file_topo  = ', TRIM(file_topo)
write(*,*) '  varnam     = ', TRIM(varnam)

!----------------------------------------------------------------
! Read CMIP6 data (previously interpolated to the ISMIP6 grid) :

write(*,*) 'Reading ', TRIM(file_in)

status = NF90_OPEN(TRIM(file_in),0,fidA); call erreur(status,.TRUE.,"read")

status = NF90_INQ_DIMID(fidA,"x",dimID_x); call erreur(status,.TRUE.,"inq_dimID_x")
status = NF90_INQ_DIMID(fidA,"y",dimID_y); call erreur(status,.TRUE.,"inq_dimID_y")
status = NF90_INQ_DIMID(fidA,"z",dimID_z); call erreur(status,.TRUE.,"inq_dimID_z")
status = NF90_INQ_DIMID(fidA,"time",dimID_time); call erreur(status,.TRUE.,"inq_dimID_time")

status = NF90_INQUIRE_DIMENSION(fidA,dimID_x,len=mx); call erreur(status,.TRUE.,"inq_dim_x")
status = NF90_INQUIRE_DIMENSION(fidA,dimID_y,len=my); call erreur(status,.TRUE.,"inq_dim_y")
status = NF90_INQUIRE_DIMENSION(fidA,dimID_z,len=mz); call erreur(status,.TRUE.,"inq_dim_z")
status = NF90_INQUIRE_DIMENSION(fidA,dimID_time,len=mtime); call erreur(status,.TRUE.,"inq_dim_time")

ALLOCATE(  time(mtime)  )
ALLOCATE(  z(mz)  )
ALLOCATE(  y(my)  )
ALLOCATE(  x(mx)  )
ALLOCATE(  var(mx,my,1,mtime)  )
ALLOCATE(  var_new(mx,my,1,mtime)  )

status = NF90_INQ_VARID(fidA,"time",time_ID); call erreur(status,.TRUE.,"inq_time_ID")
status = NF90_INQ_VARID(fidA,"z",z_ID); call erreur(status,.TRUE.,"inq_z_ID")
status = NF90_INQ_VARID(fidA,"y",y_ID); call erreur(status,.TRUE.,"inq_y_ID")
status = NF90_INQ_VARID(fidA,"x",x_ID); call erreur(status,.TRUE.,"inq_x_ID")
status = NF90_INQ_VARID(fidA,TRIM(varnam),varin_ID); call erreur(status,.TRUE.,"inq_var_ID")

! Determine missing value from input variable attributes. Prefer _FillValue, fall back to missing_value,
! otherwise use legacy default (-1.e6). Do this BEFORE creating the output file so we can propagate the
! exact value to the new variable's attributes.
status = NF90_GET_ATT(fidA,varin_ID,"_FillValue",miss)
if ( status /= NF90_NOERR ) then
  status = NF90_GET_ATT(fidA,varin_ID,"missing_value",miss)
  if ( status /= NF90_NOERR ) then
    miss = -1.e6
    write(*,*) 'Input variable has no _FillValue or missing_value attribute; using fallback miss = ', miss
  else
    write(*,*) 'Using input variable missing_value attribute: miss = ', miss
  end if
else
  write(*,*) 'Using input variable _FillValue attribute: miss = ', miss
end if

status = NF90_GET_ATT(fidA,time_ID,'calendar',cal)     ; call erreur(status,.TRUE.,"get_att1")
status = NF90_GET_ATT(fidA,time_ID,'units',uni)        ; call erreur(status,.TRUE.,"get_att2")
status = NF90_GET_ATT(fidA,NF90_GLOBAL,'history',his)  ; call erreur(status,.TRUE.,"get_att3")

status = NF90_GET_VAR(fidA,time_ID,time); call erreur(status,.TRUE.,"getvar_time")
status = NF90_GET_VAR(fidA,z_ID,z); call erreur(status,.TRUE.,"getvar_z")
status = NF90_GET_VAR(fidA,y_ID,y); call erreur(status,.TRUE.,"getvar_y")
status = NF90_GET_VAR(fidA,x_ID,x); call erreur(status,.TRUE.,"getvar_x")

!----------------------------------------------------------------
! Reading IMBIE2 basins :

write(*,*) 'Reading ', TRIM(file_basin)

status = NF90_OPEN(TRIM(file_basin),0,fidB); call erreur(status,.TRUE.,"read")

ALLOCATE(  basinNumber(mx,my)  )
ALLOCATE(  mskba(mx,my)  )

status = NF90_INQ_VARID(fidB,"basinNumber",basinNumber_ID); call erreur(status,.TRUE.,"inq_basinNumber_ID")

status = NF90_GET_VAR(fidB,basinNumber_ID,basinNumber); call erreur(status,.TRUE.,"getvar_basinNumber")

status = NF90_CLOSE(fidB); call erreur(status,.TRUE.,"close_file")


!----------------------------------------------------------------
! Read ice shelf mask and bed topographie

write(*,*) 'Reading ', TRIM(file_topo)

status = NF90_OPEN(TRIM(file_topo),0,fidT); call erreur(status,.TRUE.,"read")

ALLOCATE(  mask(mx,my)  )
ALLOCATE(  bed(mx,my)  )
ALLOCATE(  mask_ocean(mx,my)  )

status = NF90_INQ_VARID(fidT,"mask",mask_ID); call erreur(status,.TRUE.,"inq_mask_ID")
status = NF90_INQ_VARID(fidT,"bed",bed_ID); call erreur(status,.TRUE.,"inq_bed_ID")
status = NF90_INQ_VARID(fidT,"mask_ocean",mask_ocean_ID); call erreur(status,.TRUE.,"inq_mask_ocean_ID")

status = NF90_GET_VAR(fidT,mask_ID,mask); call erreur(status,.TRUE.,"getvar_mask")
status = NF90_GET_VAR(fidT,bed_ID,bed); call erreur(status,.TRUE.,"getvar_bed")
status = NF90_GET_VAR(fidT,mask_ocean_ID,mask_ocean); call erreur(status,.TRUE.,"getvar_mask_ocean")

status = NF90_CLOSE(fidT); call erreur(status,.TRUE.,"close_file")

!-------------------------------------------------------------
! Preparing new netcdf file with extrapolated data:

write(*,*) 'Creating ', TRIM(file_out)

status = NF90_CREATE(TRIM(file_out),NF90_NOCLOBBER,fidM); call erreur(status,.TRUE.,'create')

status = NF90_DEF_DIM(fidM,"x",mx,dimID_x); call erreur(status,.TRUE.,"def_dimID_x")
status = NF90_DEF_DIM(fidM,"y",my,dimID_y); call erreur(status,.TRUE.,"def_dimID_y")
status = NF90_DEF_DIM(fidM,"z",mz,dimID_z); call erreur(status,.TRUE.,"def_dimID_z")
status = NF90_DEF_DIM(fidM,"time",NF90_UNLIMITED,dimID_time); call erreur(status,.TRUE.,"def_dimID_time")

status = NF90_DEF_VAR(fidM,"time",NF90_DOUBLE,(/dimID_time/),time_ID); call erreur(status,.TRUE.,"def_var_time_ID")
status = NF90_DEF_VAR(fidM,"z",NF90_FLOAT,(/dimID_z/),z_ID); call erreur(status,.TRUE.,"def_var_z_ID")
status = NF90_DEF_VAR(fidM,"y",NF90_FLOAT,(/dimID_y/),y_ID); call erreur(status,.TRUE.,"def_var_y_ID")
status = NF90_DEF_VAR(fidM,"x",NF90_FLOAT,(/dimID_x/),x_ID); call erreur(status,.TRUE.,"def_var_x_ID")
status = NF90_DEF_VAR(fidM,TRIM(varnam),NF90_FLOAT,(/dimID_x,dimID_y,dimID_z,dimID_time/),varout_ID); call erreur(status,.TRUE.,"def_var_var_ID")

status = NF90_PUT_ATT(fidM,time_ID,"calendar",TRIM(cal)); call erreur(status,.TRUE.,"put_att_time_ID")
status = NF90_PUT_ATT(fidM,time_ID,"units",TRIM(uni)); call erreur(status,.TRUE.,"put_att_time_ID")
status = NF90_PUT_ATT(fidM,time_ID,"standard_name","time"); call erreur(status,.TRUE.,"put_att_time_ID")
status = NF90_PUT_ATT(fidM,z_ID,"positive","up"); call erreur(status,.TRUE.,"put_att_z_ID")
status = NF90_PUT_ATT(fidM,z_ID,"long_name","depth"); call erreur(status,.TRUE.,"put_att_z_ID")
status = NF90_PUT_ATT(fidM,z_ID,"units","m"); call erreur(status,.TRUE.,"put_att_z_ID")
status = NF90_PUT_ATT(fidM,y_ID,"long_name","y coordinate"); call erreur(status,.TRUE.,"put_att_y_ID")
status = NF90_PUT_ATT(fidM,y_ID,"units","m"); call erreur(status,.TRUE.,"put_att_y_ID")
status = NF90_PUT_ATT(fidM,x_ID,"long_name","x coordinate"); call erreur(status,.TRUE.,"put_att_x_ID")
status = NF90_PUT_ATT(fidM,x_ID,"units","m"); call erreur(status,.TRUE.,"put_att_x_ID")

status = NF90_PUT_ATT(fidM,varout_ID,"_FillValue",miss); call erreur(status,.TRUE.,"put_att_var__FillValue")

status = NF90_PUT_ATT(fidM,NF90_GLOBAL,"project","EU-H2020-PROTECT"); call erreur(status,.TRUE.,"att_GLO1")
status = NF90_PUT_ATT(fidM,NF90_GLOBAL,"history",TRIM(his)); call erreur(status,.TRUE.,"att_GLO2")
status = NF90_PUT_ATT(fidM,NF90_GLOBAL,"method","see https://github.com/nicojourdain/CMIP6_data_to_ISMIP6_grid"); call erreur(status,.TRUE.,"att_GLO3")

status = NF90_ENDDEF(fidM); call erreur(status,.TRUE.,"end_definition")

status = NF90_PUT_VAR(fidM,time_ID,time); call erreur(status,.TRUE.,"putvar_time")
status = NF90_PUT_VAR(fidM,z_ID,z); call erreur(status,.TRUE.,"putvar_z")
status = NF90_PUT_VAR(fidM,y_ID,y); call erreur(status,.TRUE.,"putvar_y")
status = NF90_PUT_VAR(fidM,x_ID,x); call erreur(status,.TRUE.,"putvar_x")

!---------------------------------------------------------------------------------
Nbasin=MAXVAL(basinNumber)+1
write(*,*) 'Number of IMBIE2 basins : ', Nbasin

! number of iteration to fill the ice shelves in each basin:
ALLOCATE( Niter(Nbasin) )

! 'miss' already defined from input file attributes (or fallback). Ensure any NaNs are replaced by this value.

DO kz=1,mz
  !write(*,*) '     kz ', kz

  ! read 2d variable in netcdf
  status = NF90_GET_VAR(fidA,varin_ID,var,start=(/1,1,kz,1/),count=(/mx,my,1,mtime/))
  call erreur(status,.TRUE.,"get_var")

  do ki=1,mx
  do kj=1,my
  do kt=1,mtime
    if ( ISNAN(var(ki,kj,1,kt)) )  var(ki,kj,1,kt) = miss
  enddo
  enddo
  enddo

  !=== step1: interpolation in basin and only into present-day ice shelf cavities and open ocean
  Niter(:) = 20
  Niter(3) = 75  ! Amery
  Niter(8) = 120 ! Ross
  Niter(15) = 120 ! FRIS

  do kbasin=1,Nbasin

   do kiter=1,Niter(kbasin)

    var_new(:,:,:,:) = var(:,:,:,:)

    do ki=1,mx
    do kj=1,my
  if ( var(ki,kj,1,1) /= miss .and. bed(ki,kj) .lt. z(kz) .and. basinNumber(ki,kj) .eq. kbasin-1 ) then
              mskba(ki,kj) = 1
      else
              mskba(ki,kj) = 0
      endif
    enddo
    enddo

    do ki=1,mx
    do kj=1,my
      ! open ocean or ice shelf point with missing data and connected to ocean neighbours
  if ( var(ki,kj,1,1) == miss .and. bed(ki,kj) .lt. z(kz) .and. basinNumber(ki,kj) .eq. kbasin-1 &
  &    .and. ( mask(ki,kj) .gt. 0.01 .or. mask_ocean(ki,kj) .gt. 0.01 ) ) then

           ! Gaussian (sigma=24km) extrapolation to contiguous neighbours (needs continuous connection to (ki,kj) ) :

           kim1=MAX(ki-1, 1) ; kim2=MAX(ki-2, 1) ; kim3=MAX(ki-3, 1)
           kip1=MIN(ki+1,mx) ; kip2=MIN(ki+2,mx) ; kip3=MIN(ki+3,mx)
           kjm1=MAX(kj-1, 1) ; kjm2=MAX(kj-2, 1) ; kjm3=MAX(kj-3, 1)
           kjp1=MIN(kj+1,my) ; kjp2=MIN(kj+2,my) ; kjp3=MIN(kj+3,my)

           bbim1jjj = mskba(kim1,kj) * 1.e0
           bbip1jjj = mskba(kip1,kj) * 1.e0
           bbiiijm1 = mskba(ki,kjm1) * 1.e0
           bbiiijp1 = mskba(ki,kjp1) * 1.e0
           !-
           bbim2jjj = bbim1jjj * mskba(kim2,kj)
           bbip2jjj = bbim1jjj * mskba(kip2,kj)
           bbiiijm2 = bbiiijm1 * mskba(ki,kjm2)
           bbiiijp2 = bbiiijm1 * mskba(ki,kjp2)
           !-
           bbim1jm1 = MAX(bbim1jjj,bbiiijm1) * mskba(kim1,kjm1)
           bbim1jp1 = MAX(bbim1jjj,bbiiijp1) * mskba(kim1,kjp1)
           bbip1jm1 = MAX(bbip1jjj,bbiiijm1) * mskba(kip1,kjm1)
           bbip1jp1 = MAX(bbip1jjj,bbiiijp1) * mskba(kip1,kjp1)
           !-
           bbim2jp1 = MAX(bbim2jjj,bbim1jp1) * mskba(kim2,kjp1)
           bbim2jm1 = MAX(bbim2jjj,bbim1jm1) * mskba(kim2,kjm1)
           bbip2jp1 = MAX(bbip2jjj,bbip1jp1) * mskba(kip2,kjp1)
           bbip2jm1 = MAX(bbip2jjj,bbip1jm1) * mskba(kip2,kjm1)
           bbim1jp2 = MAX(bbim1jp1,bbiiijp2) * mskba(kim1,kjp2)
           bbip1jp2 = MAX(bbip1jp1,bbiiijp2) * mskba(kip1,kjp2)
           bbim1jm2 = MAX(bbim1jm1,bbiiijm2) * mskba(kim1,kjm2)
           bbip1jm2 = MAX(bbip1jm1,bbiiijm2) * mskba(kip1,kjm2)
           !-
           bbim2jm2 = MAX(bbim2jm1,bbim1jm2) * mskba(kim2,kjm2)
           bbim2jp2 = MAX(bbim2jp1,bbim1jp2) * mskba(kim2,kjp2)
           bbip2jm2 = MAX(bbip2jm1,bbip1jm2) * mskba(kip2,kjm2)
           bbip2jp2 = MAX(bbip2jp1,bbip1jp2) * mskba(kip2,kjp2)
           !-
           bbim3jjj = bbim2jjj * mskba(kim3,kj)
           bbip3jjj = bbim2jjj * mskba(kip3,kj)
           bbiiijm3 = bbiiijm2 * mskba(ki,kjm3)
           bbiiijp3 = bbiiijm2 * mskba(ki,kjp3)

           bb =   ( bbim1jjj + bbip1jjj + bbiiijm1 + bbiiijp1 ) * 0.946 & ! normalised Gaussian of sigma=24km at x=8km
           &    + ( bbim2jjj + bbip2jjj + bbiiijm2 + bbiiijp2 ) * 0.801 & !  "     "      "       "       "      x=16km
           &    + ( bbim1jm1 + bbim1jp1 + bbip1jm1 + bbip1jp1 ) * 0.895 & !  "     "      "       "       "      x=11.3km
           &    + ( bbim2jp1 + bbim2jm1 + bbip2jp1 + bbip2jm1 + bbim1jp2 + bbip1jp2 + bbim1jm2 + bbip1jm2 ) * 0.757 &  ! " x=17.9km
           &    + ( bbim2jm2 + bbim2jp2 + bbip2jm2 + bbip2jp2 ) * 0.641 & !  "     "      "       "       "      x=22.6km
           &    + ( bbim3jjj + bbip3jjj + bbiiijm3 + bbiiijp3 ) * 0.606   !  "     "      "       "       "      x=24km

           if ( bb .gt. 0.1 ) then

             do kt=1,mtime

               aa =   (  bbim1jjj*var(kim1,kj  ,1,kt) + bbip1jjj*var(kip1,kj  ,1,kt) + bbiiijm1*var(ki  ,kjm1,1,kt) + bbiiijp1*var(ki  ,kjp1,1,kt) ) * 0.946 &
               &    + (  bbim2jjj*var(kim2,kj  ,1,kt) + bbip2jjj*var(kip2,kj  ,1,kt) + bbiiijm2*var(ki  ,kjm2,1,kt) + bbiiijp2*var(ki  ,kjp2,1,kt) ) * 0.801 &
               &    + (  bbim1jm1*var(kim1,kjm1,1,kt) + bbim1jp1*var(kim1,kjp1,1,kt) + bbip1jm1*var(kip1,kjm1,1,kt) + bbip1jp1*var(kip1,kjp1,1,kt) ) * 0.895 &
               &    + (  bbim2jp1*var(kim2,kjp1,1,kt) + bbim2jm1*var(kim2,kjm1,1,kt) + bbip2jp1*var(kip2,kjp1,1,kt) + bbip2jm1*var(kip2,kjm1,1,kt)           &
               &       + bbim1jp2*var(kim1,kjp2,1,kt) + bbip1jp2*var(kip1,kjp2,1,kt) + bbim1jm2*var(kim1,kjm2,1,kt) + bbip1jm2*var(kip1,kjm2,1,kt) ) * 0.757 &
               &    + (  bbim2jm2*var(kim2,kjm2,1,kt) + bbim2jp2*var(kim2,kjp2,1,kt) + bbip2jm2*var(kip2,kjm2,1,kt) + bbip2jp2*var(kip2,kjp2,1,kt) ) * 0.641 &
               &    + (  bbim3jjj*var(kim3,kj  ,1,kt) + bbip3jjj*var(kip3,kj  ,1,kt) + bbiiijm3*var(ki  ,kjm3,1,kt) + bbiiijp3*var(ki  ,kjp3,1,kt) ) * 0.606

               var_new(ki,kj,1,kt) = aa / bb

             enddo

           endif
      endif
    enddo
    enddo

    var(:,:,:,:) = var_new(:,:,:,:)

   enddo ! iterations

  enddo ! kbasin

  !=== step2: horizontal interpolation beyond present-day ice shelves (where bathy allows):
  Niter(:) = 120
  Niter(5:9) = 200 ! 8=Ross
  Niter(11:12) = 40
  Niter(14) = 40
  Niter(15) = 200 ! FRIS
  Niter(16) = 40

  do kbasin=1,Nbasin

   do kiter=1,Niter(kbasin)

    var_new(:,:,:,:) = var(:,:,:,:)

    do ki=1,mx
    do kj=1,my
  if ( var(ki,kj,1,1) /= miss .and. bed(ki,kj) .lt. z(kz) .and. basinNumber(ki,kj) .eq. kbasin-1 ) then
              mskba(ki,kj) = 1
      else
              mskba(ki,kj) = 0
      endif
    enddo
    enddo

    do ki=1,mx
    do kj=1,my
      ! open ocean or ice shelf point with missing data and connected to ocean neighbours
  if ( var(ki,kj,1,1) == miss .and. bed(ki,kj) .lt. z(kz) .and. basinNumber(ki,kj) .eq. kbasin-1 ) then

           ! Gaussian (sigma=24km) extrapolation to contiguous neighbours (needs continuous connection to (ki,kj) ) :

           kim1=MAX(ki-1, 1) ; kim2=MAX(ki-2, 1) ; kim3=MAX(ki-3, 1)
           kip1=MIN(ki+1,mx) ; kip2=MIN(ki+2,mx) ; kip3=MIN(ki+3,mx)
           kjm1=MAX(kj-1, 1) ; kjm2=MAX(kj-2, 1) ; kjm3=MAX(kj-3, 1)
           kjp1=MIN(kj+1,my) ; kjp2=MIN(kj+2,my) ; kjp3=MIN(kj+3,my)

           bbim1jjj = mskba(kim1,kj) * 1.e0
           bbip1jjj = mskba(kip1,kj) * 1.e0
           bbiiijm1 = mskba(ki,kjm1) * 1.e0
           bbiiijp1 = mskba(ki,kjp1) * 1.e0
           !-
           bbim2jjj = bbim1jjj * mskba(kim2,kj)
           bbip2jjj = bbim1jjj * mskba(kip2,kj)
           bbiiijm2 = bbiiijm1 * mskba(ki,kjm2)
           bbiiijp2 = bbiiijm1 * mskba(ki,kjp2)
           !-
           bbim1jm1 = MAX(bbim1jjj,bbiiijm1) * mskba(kim1,kjm1)
           bbim1jp1 = MAX(bbim1jjj,bbiiijp1) * mskba(kim1,kjp1)
           bbip1jm1 = MAX(bbip1jjj,bbiiijm1) * mskba(kip1,kjm1)
           bbip1jp1 = MAX(bbip1jjj,bbiiijp1) * mskba(kip1,kjp1)
           !-
           bbim2jp1 = MAX(bbim2jjj,bbim1jp1) * mskba(kim2,kjp1)
           bbim2jm1 = MAX(bbim2jjj,bbim1jm1) * mskba(kim2,kjm1)
           bbip2jp1 = MAX(bbip2jjj,bbip1jp1) * mskba(kip2,kjp1)
           bbip2jm1 = MAX(bbip2jjj,bbip1jm1) * mskba(kip2,kjm1)
           bbim1jp2 = MAX(bbim1jp1,bbiiijp2) * mskba(kim1,kjp2)
           bbip1jp2 = MAX(bbip1jp1,bbiiijp2) * mskba(kip1,kjp2)
           bbim1jm2 = MAX(bbim1jm1,bbiiijm2) * mskba(kim1,kjm2)
           bbip1jm2 = MAX(bbip1jm1,bbiiijm2) * mskba(kip1,kjm2)
           !-
           bbim2jm2 = MAX(bbim2jm1,bbim1jm2) * mskba(kim2,kjm2)
           bbim2jp2 = MAX(bbim2jp1,bbim1jp2) * mskba(kim2,kjp2)
           bbip2jm2 = MAX(bbip2jm1,bbip1jm2) * mskba(kip2,kjm2)
           bbip2jp2 = MAX(bbip2jp1,bbip1jp2) * mskba(kip2,kjp2)
           !-
           bbim3jjj = bbim2jjj * mskba(kim3,kj)
           bbip3jjj = bbim2jjj * mskba(kip3,kj)
           bbiiijm3 = bbiiijm2 * mskba(ki,kjm3)
           bbiiijp3 = bbiiijm2 * mskba(ki,kjp3)

           bb =   ( bbim1jjj + bbip1jjj + bbiiijm1 + bbiiijp1 ) * 0.946 & ! normalised Gaussian of sigma=24km at x=8km
           &    + ( bbim2jjj + bbip2jjj + bbiiijm2 + bbiiijp2 ) * 0.801 & !  "     "      "       "       "      x=16km
           &    + ( bbim1jm1 + bbim1jp1 + bbip1jm1 + bbip1jp1 ) * 0.895 & !  "     "      "       "       "      x=11.3km
           &    + ( bbim2jp1 + bbim2jm1 + bbip2jp1 + bbip2jm1 + bbim1jp2 + bbip1jp2 + bbim1jm2 + bbip1jm2 ) * 0.757 &  ! " x=17.9km
           &    + ( bbim2jm2 + bbim2jp2 + bbip2jm2 + bbip2jp2 ) * 0.641 & !  "     "      "       "       "      x=22.6km
           &    + ( bbim3jjj + bbip3jjj + bbiiijm3 + bbiiijp3 ) * 0.606   !  "     "      "       "       "      x=24km

           if ( bb .gt. 0.1 ) then

             do kt=1,mtime

               aa =   (  bbim1jjj*var(kim1,kj  ,1,kt) + bbip1jjj*var(kip1,kj  ,1,kt) + bbiiijm1*var(ki  ,kjm1,1,kt) + bbiiijp1*var(ki  ,kjp1,1,kt) ) * 0.946 &
               &    + (  bbim2jjj*var(kim2,kj  ,1,kt) + bbip2jjj*var(kip2,kj  ,1,kt) + bbiiijm2*var(ki  ,kjm2,1,kt) + bbiiijp2*var(ki  ,kjp2,1,kt) ) * 0.801 &
               &    + (  bbim1jm1*var(kim1,kjm1,1,kt) + bbim1jp1*var(kim1,kjp1,1,kt) + bbip1jm1*var(kip1,kjm1,1,kt) + bbip1jp1*var(kip1,kjp1,1,kt) ) * 0.895 &
               &    + (  bbim2jp1*var(kim2,kjp1,1,kt) + bbim2jm1*var(kim2,kjm1,1,kt) + bbip2jp1*var(kip2,kjp1,1,kt) + bbip2jm1*var(kip2,kjm1,1,kt)           &
               &       + bbim1jp2*var(kim1,kjp2,1,kt) + bbip1jp2*var(kip1,kjp2,1,kt) + bbim1jm2*var(kim1,kjm2,1,kt) + bbip1jm2*var(kip1,kjm2,1,kt) ) * 0.757 &
               &    + (  bbim2jm2*var(kim2,kjm2,1,kt) + bbim2jp2*var(kim2,kjp2,1,kt) + bbip2jm2*var(kip2,kjm2,1,kt) + bbip2jp2*var(kip2,kjp2,1,kt) ) * 0.641 &
               &    + (  bbim3jjj*var(kim3,kj  ,1,kt) + bbip3jjj*var(kip3,kj  ,1,kt) + bbiiijm3*var(ki  ,kjm3,1,kt) + bbiiijp3*var(ki  ,kjp3,1,kt) ) * 0.606

               var_new(ki,kj,1,kt) = aa / bb

             enddo

           endif
      endif
    enddo
    enddo

    var(:,:,:,:) = var_new(:,:,:,:)

   enddo ! iterations

  enddo ! kbasin

  !=== step3: horizontal interpolation everywhere AT THE SURFACE (k=2), not accounting for basins
  !           (even if bedrock above sea level, to extrapolate downward everywhere later on).
  !           AND extrapolation everywhere of 5 grid points to be less sensitive to BedMachine values
  !           but not to fill deep isolated holes like Amery.
  if ( kz.eq. 2) then ! takes 2nd level to avoid warn thin surface layer at the surface
    Niter2=200 ! fills entire level
  else
    Niter2=5
  endif

  do kiter=1,Niter2

      var_new(:,:,:,:) = var(:,:,:,:)

      do ki=1,mx
      do kj=1,my
  if ( var(ki,kj,1,1) /= miss ) then
                mskba(ki,kj) = 1
        else
                mskba(ki,kj) = 0
        endif
      enddo
      enddo

      do ki=1,mx
      do kj=1,my
        ! open ocean or ice shelf point with missing data and connected to ocean neighbours
  if ( var(ki,kj,1,1) == miss ) then

             ! Gaussian (sigma=24km) extrapolation to contiguous neighbours (needs continuous connection to (ki,kj) ) :

             kim1=MAX(ki-1, 1) ; kim2=MAX(ki-2, 1) ; kim3=MAX(ki-3, 1)
             kip1=MIN(ki+1,mx) ; kip2=MIN(ki+2,mx) ; kip3=MIN(ki+3,mx)
             kjm1=MAX(kj-1, 1) ; kjm2=MAX(kj-2, 1) ; kjm3=MAX(kj-3, 1)
             kjp1=MIN(kj+1,my) ; kjp2=MIN(kj+2,my) ; kjp3=MIN(kj+3,my)

             bbim1jjj = mskba(kim1,kj) * 1.e0
             bbip1jjj = mskba(kip1,kj) * 1.e0
             bbiiijm1 = mskba(ki,kjm1) * 1.e0
             bbiiijp1 = mskba(ki,kjp1) * 1.e0
             !-
             bbim2jjj = bbim1jjj * mskba(kim2,kj)
             bbip2jjj = bbim1jjj * mskba(kip2,kj)
             bbiiijm2 = bbiiijm1 * mskba(ki,kjm2)
             bbiiijp2 = bbiiijm1 * mskba(ki,kjp2)
             !-
             bbim1jm1 = MAX(bbim1jjj,bbiiijm1) * mskba(kim1,kjm1)
             bbim1jp1 = MAX(bbim1jjj,bbiiijp1) * mskba(kim1,kjp1)
             bbip1jm1 = MAX(bbip1jjj,bbiiijm1) * mskba(kip1,kjm1)
             bbip1jp1 = MAX(bbip1jjj,bbiiijp1) * mskba(kip1,kjp1)
             !-
             bbim2jp1 = MAX(bbim2jjj,bbim1jp1) * mskba(kim2,kjp1)
             bbim2jm1 = MAX(bbim2jjj,bbim1jm1) * mskba(kim2,kjm1)
             bbip2jp1 = MAX(bbip2jjj,bbip1jp1) * mskba(kip2,kjp1)
             bbip2jm1 = MAX(bbip2jjj,bbip1jm1) * mskba(kip2,kjm1)
             bbim1jp2 = MAX(bbim1jp1,bbiiijp2) * mskba(kim1,kjp2)
             bbip1jp2 = MAX(bbip1jp1,bbiiijp2) * mskba(kip1,kjp2)
             bbim1jm2 = MAX(bbim1jm1,bbiiijm2) * mskba(kim1,kjm2)
             bbip1jm2 = MAX(bbip1jm1,bbiiijm2) * mskba(kip1,kjm2)
             !-
             bbim2jm2 = MAX(bbim2jm1,bbim1jm2) * mskba(kim2,kjm2)
             bbim2jp2 = MAX(bbim2jp1,bbim1jp2) * mskba(kim2,kjp2)
             bbip2jm2 = MAX(bbip2jm1,bbip1jm2) * mskba(kip2,kjm2)
             bbip2jp2 = MAX(bbip2jp1,bbip1jp2) * mskba(kip2,kjp2)
             !-
             bbim3jjj = bbim2jjj * mskba(kim3,kj)
             bbip3jjj = bbim2jjj * mskba(kip3,kj)
             bbiiijm3 = bbiiijm2 * mskba(ki,kjm3)
             bbiiijp3 = bbiiijm2 * mskba(ki,kjp3)

             bb =   ( bbim1jjj + bbip1jjj + bbiiijm1 + bbiiijp1 ) * 0.946 & ! normalised Gaussian of sigma=24km at x=8km
             &    + ( bbim2jjj + bbip2jjj + bbiiijm2 + bbiiijp2 ) * 0.801 & !  "     "      "       "       "      x=16km
             &    + ( bbim1jm1 + bbim1jp1 + bbip1jm1 + bbip1jp1 ) * 0.895 & !  "     "      "       "       "      x=11.3km
             &    + ( bbim2jp1 + bbim2jm1 + bbip2jp1 + bbip2jm1 + bbim1jp2 + bbip1jp2 + bbim1jm2 + bbip1jm2 ) * 0.757 &  ! " x=17.9km
             &    + ( bbim2jm2 + bbim2jp2 + bbip2jm2 + bbip2jp2 ) * 0.641 & !  "     "      "       "       "      x=22.6km
             &    + ( bbim3jjj + bbip3jjj + bbiiijm3 + bbiiijp3 ) * 0.606   !  "     "      "       "       "      x=24km

             if ( bb .gt. 0.1 ) then

               do kt=1,mtime

                 aa =   (  bbim1jjj*var(kim1,kj  ,1,kt) + bbip1jjj*var(kip1,kj  ,1,kt) + bbiiijm1*var(ki  ,kjm1,1,kt) + bbiiijp1*var(ki  ,kjp1,1,kt) ) * 0.946 &
                 &    + (  bbim2jjj*var(kim2,kj  ,1,kt) + bbip2jjj*var(kip2,kj  ,1,kt) + bbiiijm2*var(ki  ,kjm2,1,kt) + bbiiijp2*var(ki  ,kjp2,1,kt) ) * 0.801 &
                 &    + (  bbim1jm1*var(kim1,kjm1,1,kt) + bbim1jp1*var(kim1,kjp1,1,kt) + bbip1jm1*var(kip1,kjm1,1,kt) + bbip1jp1*var(kip1,kjp1,1,kt) ) * 0.895 &
                 &    + (  bbim2jp1*var(kim2,kjp1,1,kt) + bbim2jm1*var(kim2,kjm1,1,kt) + bbip2jp1*var(kip2,kjp1,1,kt) + bbip2jm1*var(kip2,kjm1,1,kt)           &
                 &       + bbim1jp2*var(kim1,kjp2,1,kt) + bbip1jp2*var(kip1,kjp2,1,kt) + bbim1jm2*var(kim1,kjm2,1,kt) + bbip1jm2*var(kip1,kjm2,1,kt) ) * 0.757 &
                 &    + (  bbim2jm2*var(kim2,kjm2,1,kt) + bbim2jp2*var(kim2,kjp2,1,kt) + bbip2jm2*var(kip2,kjm2,1,kt) + bbip2jp2*var(kip2,kjp2,1,kt) ) * 0.641 &
                 &    + (  bbim3jjj*var(kim3,kj  ,1,kt) + bbip3jjj*var(kip3,kj  ,1,kt) + bbiiijm3*var(ki  ,kjm3,1,kt) + bbiiijp3*var(ki  ,kjp3,1,kt) ) * 0.606

                 var_new(ki,kj,1,kt) = aa / bb

               enddo

             endif
        endif
      enddo
      enddo

      var(:,:,:,:) = var_new(:,:,:,:)

  enddo ! iterations

  !== write variable in netcdf
  status = NF90_PUT_VAR(fidM,varout_ID,var,start=(/1,1,kz,1/),count=(/mx,my,1,mtime/))
  call erreur(status,.TRUE.,"put_var")

ENDDO ! kz

!---------------------

status = NF90_CLOSE(fidM); call erreur(status,.TRUE.,"close_M")
status = NF90_CLOSE(fidA); call erreur(status,.TRUE.,"close_A")

end program modif


!=====================================================================================
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
