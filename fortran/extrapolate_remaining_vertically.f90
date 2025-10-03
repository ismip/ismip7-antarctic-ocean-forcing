program modif

USE netcdf

IMPLICIT NONE

INTEGER :: fidA, status, dimID_x, dimID_y, dimID_z, dimID_time, mx, my, mz, mtime, time_ID, &
&          z_ID, y_ID, x_ID, var_in_ID, var_out_ID, fidM, ki, kj, kz, kt

CHARACTER(LEN=15) :: varnam

CHARACTER(LEN=2048) :: file_in, file_out, cal, uni, his
CHARACTER(LEN=2048) :: namelist_file
CHARACTER(LEN=32)   :: z_name

INTEGER :: ios, narg

NAMELIST /vertical_extrapolation/ file_in, file_out, varnam, z_name

REAL*4,ALLOCATABLE,DIMENSION(:) :: z, y, x

REAL*8,ALLOCATABLE,DIMENSION(:) :: time

REAL*4,ALLOCATABLE,DIMENSION(:,:,:) :: var_in
REAL*4 :: miss

!---------------------------------------
! Namelist-driven configuration with invalid defaults to force user input
file_in  = '__REQUIRED__'
file_out = '__REQUIRED__'
varnam   = '__REQUIRED__'
z_name   = 'z'

! Determine namelist file (first command-line argument or default name)
narg = COMMAND_ARGUMENT_COUNT()
if ( narg >= 1 ) then
  call GET_COMMAND_ARGUMENT(1, namelist_file, status=ios)
  if ( ios /= 0 ) then
    write(*,*) 'ERROR retrieving command argument for namelist file.'
    stop 1
  end if
  if ( LEN_TRIM(namelist_file) == 0 ) namelist_file = 'extrapolate_vertical.nml'
else
  namelist_file = 'extrapolate_vertical.nml'
end if

open(unit=31, file=TRIM(namelist_file), status='old', action='read', iostat=ios)
if ( ios /= 0 ) then
  write(*,*) 'ERROR: cannot open namelist file: ', TRIM(namelist_file)
  stop 1
end if
read(31, nml=vertical_extrapolation, iostat=ios)
if ( ios /= 0 ) then
  write(*,*) 'ERROR: reading namelist "vertical_extrapolation" from file: ', TRIM(namelist_file)
  stop 1
end if
close(31)

if ( TRIM(file_in)  == '__REQUIRED__' .or. &
    TRIM(file_out) == '__REQUIRED__' .or. &
    TRIM(varnam)   == '__REQUIRED__' ) then
  write(*,*) 'ERROR: One or more required namelist entries missing.'
  write(*,*) '       file_in  = ', TRIM(file_in)
  write(*,*) '       file_out = ', TRIM(file_out)
  write(*,*) '       varnam   = ', TRIM(varnam)
  stop 1
end if

write(*,*) 'Using namelist: ', TRIM(namelist_file)
write(*,*) '  file_in  = ', TRIM(file_in)
write(*,*) '  file_out = ', TRIM(file_out)
write(*,*) '  varnam   = ', TRIM(varnam)

!---------------------------------------
! Read netcdf input file :

write(*,*) 'Reading ', TRIM(file_in)

status = NF90_OPEN(TRIM(file_in),0,fidA); call erreur(status,.TRUE.,"read")

status = NF90_INQ_DIMID(fidA,"x",dimID_x); call erreur(status,.TRUE.,"inq_dimID_x")
status = NF90_INQ_DIMID(fidA,"y",dimID_y); call erreur(status,.TRUE.,"inq_dimID_y")
status = NF90_INQ_DIMID(fidA,TRIM(z_name),dimID_z); call erreur(status,.TRUE.,"inq_dimID_z")
status = NF90_INQ_DIMID(fidA,"time",dimID_time); call erreur(status,.TRUE.,"inq_dimID_time")

status = NF90_INQUIRE_DIMENSION(fidA,dimID_x,len=mx); call erreur(status,.TRUE.,"inq_dim_x")
status = NF90_INQUIRE_DIMENSION(fidA,dimID_y,len=my); call erreur(status,.TRUE.,"inq_dim_y")
status = NF90_INQUIRE_DIMENSION(fidA,dimID_z,len=mz); call erreur(status,.TRUE.,"inq_dim_z")
status = NF90_INQUIRE_DIMENSION(fidA,dimID_time,len=mtime); call erreur(status,.TRUE.,"inq_dim_time")

ALLOCATE(  time(mtime)  )
ALLOCATE(  z(mz)  )
ALLOCATE(  y(my)  )
ALLOCATE(  x(mx)  )
ALLOCATE(  var_in(mx,my,mz)  )

status = NF90_INQ_VARID(fidA,"time",time_ID); call erreur(status,.TRUE.,"inq_time_ID")
status = NF90_INQ_VARID(fidA,TRIM(z_name),z_ID); call erreur(status,.TRUE.,"inq_z_ID")
status = NF90_INQ_VARID(fidA,"y",y_ID); call erreur(status,.TRUE.,"inq_y_ID")
status = NF90_INQ_VARID(fidA,"x",x_ID); call erreur(status,.TRUE.,"inq_x_ID")
status = NF90_INQ_VARID(fidA,TRIM(varnam),var_in_ID); call erreur(status,.TRUE.,"inq_var_ID")

! Determine missing value from input variable; prefer _FillValue then missing_value; fallback if absent
status = NF90_GET_ATT(fidA,var_in_ID,"_FillValue",miss)
if ( status /= NF90_NOERR ) then
  status = NF90_GET_ATT(fidA,var_in_ID,"missing_value",miss)
  if ( status /= NF90_NOERR ) then
    miss = -1.e6
    write(*,*) 'Input variable has no _FillValue or missing_value; using fallback miss = ', miss
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
!status = NF90_GET_VAR(fidA,var_in_ID,var_in); call erreur(status,.TRUE.,"getvar_var")

!---------------------------------------
! Writing new netcdf file :

write(*,*) 'Creating ', TRIM(file_out)

status = NF90_CREATE(TRIM(file_out),NF90_NOCLOBBER,fidM); call erreur(status,.TRUE.,'create')

status = NF90_DEF_DIM(fidM,"x",mx,dimID_x); call erreur(status,.TRUE.,"def_dimID_x")
status = NF90_DEF_DIM(fidM,"y",my,dimID_y); call erreur(status,.TRUE.,"def_dimID_y")
status = NF90_DEF_DIM(fidM,TRIM(z_name),mz,dimID_z); call erreur(status,.TRUE.,"def_dimID_z")
status = NF90_DEF_DIM(fidM,"time",NF90_UNLIMITED,dimID_time); call erreur(status,.TRUE.,"def_dimID_time")

status = NF90_DEF_VAR(fidM,"time",NF90_DOUBLE,(/dimID_time/),time_ID); call erreur(status,.TRUE.,"def_var_time_ID")
status = NF90_DEF_VAR(fidM,TRIM(z_name),NF90_FLOAT,(/dimID_z/),z_ID); call erreur(status,.TRUE.,"def_var_z_ID")
status = NF90_DEF_VAR(fidM,"y",NF90_FLOAT,(/dimID_y/),y_ID); call erreur(status,.TRUE.,"def_var_y_ID")
status = NF90_DEF_VAR(fidM,"x",NF90_FLOAT,(/dimID_x/),x_ID); call erreur(status,.TRUE.,"def_var_x_ID")
status = NF90_DEF_VAR(fidM,TRIM(varnam),NF90_FLOAT,(/dimID_x,dimID_y,dimID_z,dimID_time/),var_out_ID); call erreur(status,.TRUE.,"def_var_var_ID")

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

status = NF90_PUT_ATT(fidM,var_out_ID,"_FillValue",miss); call erreur(status,.TRUE.,"put_att_var__FillValue")

status = NF90_PUT_ATT(fidM,NF90_GLOBAL,"project","EU-H2020-PROTECT"); call erreur(status,.TRUE.,"att_GLO1")
status = NF90_PUT_ATT(fidM,NF90_GLOBAL,"history",TRIM(his)); call erreur(status,.TRUE.,"att_GLO2")
status = NF90_PUT_ATT(fidM,NF90_GLOBAL,"method","see https://github.com/nicojourdain/CMIP6_data_to_ISMIP6_grid"); call erreur(status,.TRUE.,"att_GLO3")

status = NF90_ENDDEF(fidM); call erreur(status,.TRUE.,"fin_definition")

status = NF90_PUT_VAR(fidM,time_ID,time); call erreur(status,.TRUE.,"var_time_ID")
status = NF90_PUT_VAR(fidM,z_ID,z); call erreur(status,.TRUE.,"var_z_ID")
status = NF90_PUT_VAR(fidM,y_ID,y); call erreur(status,.TRUE.,"var_y_ID")
status = NF90_PUT_VAR(fidM,x_ID,x); call erreur(status,.TRUE.,"var_x_ID")

!----------------------------------------------------------------------------------------
! Extrapolating vertically from kz=2 :
!
DO kt=1,mtime

  write(*,*) 'Vertical: processing time index kt = ', kt, ' of ', mtime

  status = NF90_GET_VAR(fidA,var_in_ID,var_in,start=(/1,1,1,kt/),count=(/mx,my,mz,1/))
  call erreur(status,.TRUE.,"getvar_in")

  ! Replace any NaNs with miss first (defensive; should be rare if upstream used fill values)
  do ki=1,mx
  do kj=1,my
    do kz=1,mz
      if ( ISNAN(var_in(ki,kj,kz)) ) var_in(ki,kj,kz) = miss
    enddo
  enddo
  enddo

  ! Fill surface (k=1) from k=2 where k=1 is missing and k=2 valid
  do ki=1,mx
  do kj=1,my
    if ( var_in(ki,kj,1) == miss .and. var_in(ki,kj,2) /= miss ) then
       var_in(ki,kj,1) = var_in(ki,kj,2)
    endif
  enddo
  enddo

  DO kz=3,mz
    do ki=1,mx
    do kj=1,my
      if ( var_in(ki,kj,kz) == miss .and. var_in(ki,kj,kz-1) /= miss ) then
         var_in(ki,kj,kz) = var_in(ki,kj,kz-1)
      endif
    enddo
    enddo
  ENDDO

  status = NF90_PUT_VAR(fidM,var_out_ID,var_in,start=(/1,1,1,kt/),count=(/mx,my,mz,1/))
  call erreur(status,.TRUE.,"putvar_out")

ENDDO

!----------------------------------------------------------------------------------------

status = NF90_CLOSE(fidA); call erreur(status,.TRUE.,"close_file")
status = NF90_CLOSE(fidM); call erreur(status,.TRUE.,"final")

end program modif



SUBROUTINE erreur(iret, lstop, context)
  ! Error reporting helper: prints NetCDF error details.
  ! If lstop is true, terminates the program with a non-zero exit code.
  USE netcdf
  INTEGER, INTENT(in)                  :: iret
  LOGICAL, INTENT(in)                  :: lstop
  CHARACTER(LEN=*), INTENT(in)         :: context
  CHARACTER(LEN=256)                   :: message

  IF ( iret .NE. 0 ) THEN
    WRITE(*,*) 'ROUTINE: ', TRIM(context)
    WRITE(*,*) 'ERROR code: ', iret
    message = NF90_STRERROR(iret)
    WRITE(*,*) 'NETCDF message: ', TRIM(message)
    IF ( lstop ) THEN
      STOP 1
    END IF
  END IF

END SUBROUTINE erreur
