program modif
 
USE netcdf
 
IMPLICIT NONE
 
INTEGER :: fidA, status, dimID_x, dimID_y, dimID_z, dimID_time, mx, my, mz, mtime, time_ID, &
&          z_ID, y_ID, x_ID, varin_ID, varout_ID, fidM, ki, kj, kz, var_stdin_ID, mbnds,   &
&          dimID_bnds, var_stdout_ID, x_bnds_ID, y_bnds_ID, z_bnds_ID

CHARACTER(LEN=10) :: varnam

CHARACTER(LEN=150) :: file_in, file_out, uni, longnam, longnamstd, comm

CHARACTER(LEN=300) :: desc, his
 
REAL*8,ALLOCATABLE,DIMENSION(:) :: z, y, x

REAL*8,ALLOCATABLE,DIMENSION(:,:) :: z_bnds, y_bnds, x_bnds
 
REAL*4,ALLOCATABLE,DIMENSION(:,:,:) :: varin, var_stdin

REAL*4 :: miss
 
!---------------------------------------
file_in  = 'tmp_hor.nc'
file_out = '<file_out>'
 
varnam = '<var_name>'

miss = 1.e20

!---------------------------------------
! Read netcdf input file :
 
write(*,*) 'Reading ', TRIM(file_in)
 
status = NF90_OPEN(TRIM(file_in),0,fidA); call erreur(status,.TRUE.,"read")

status = NF90_INQ_DIMID(fidA,"x",dimID_x); call erreur(status,.TRUE.,"inq_dimID_x")
status = NF90_INQ_DIMID(fidA,"y",dimID_y); call erreur(status,.TRUE.,"inq_dimID_y")
status = NF90_INQ_DIMID(fidA,"z",dimID_z); call erreur(status,.TRUE.,"inq_dimID_z")
status = NF90_INQ_DIMID(fidA,"bnds",dimID_bnds); call erreur(status,.TRUE.,"inq_dimID_bnds")

status = NF90_INQUIRE_DIMENSION(fidA,dimID_x,len=mx); call erreur(status,.TRUE.,"inq_dim_x")
status = NF90_INQUIRE_DIMENSION(fidA,dimID_y,len=my); call erreur(status,.TRUE.,"inq_dim_y")
status = NF90_INQUIRE_DIMENSION(fidA,dimID_z,len=mz); call erreur(status,.TRUE.,"inq_dim_z")
status = NF90_INQUIRE_DIMENSION(fidA,dimID_bnds,len=mbnds); call erreur(status,.TRUE.,"inq_dim_bnds")

ALLOCATE(  z(mz)  )
ALLOCATE(  z_bnds(mbnds,mz)  )
ALLOCATE(  y(my)  )
ALLOCATE(  y_bnds(mbnds,my)  )
ALLOCATE(  x(mx)  )
ALLOCATE(  x_bnds(mbnds,mx)  ) 
ALLOCATE(  varin(mx,my,mz)  ) 
ALLOCATE(  var_stdin(mx,my,mz)  ) 
 
status = NF90_INQ_VARID(fidA,"z",z_ID); call erreur(status,.TRUE.,"inq_z_ID")
status = NF90_INQ_VARID(fidA,"z_bnds",z_bnds_ID); call erreur(status,.TRUE.,"inq_z_bnds_ID")
status = NF90_INQ_VARID(fidA,"y",y_ID); call erreur(status,.TRUE.,"inq_y_ID")
status = NF90_INQ_VARID(fidA,"y_bnds",y_bnds_ID); call erreur(status,.TRUE.,"inq_y_bnds_ID")
status = NF90_INQ_VARID(fidA,"x",x_ID); call erreur(status,.TRUE.,"inq_x_ID")
status = NF90_INQ_VARID(fidA,"x_bnds",x_bnds_ID); call erreur(status,.TRUE.,"inq_x_bnds_ID")
status = NF90_INQ_VARID(fidA,TRIM(varnam),varin_ID); call erreur(status,.TRUE.,"inq_var_ID")
status = NF90_INQ_VARID(fidA,TRIM(varnam)//'_uncert',var_stdin_ID); call erreur(status,.TRUE.,"inq_var_std_ID")

status = NF90_GET_ATT(fidA,varin_ID,'units',uni) ; call erreur(status,.TRUE.,"get_att1")
status = NF90_GET_ATT(fidA,varin_ID,'long_name',longnam) ; call erreur(status,.TRUE.,"get_att2")
status = NF90_GET_ATT(fidA,var_stdin_ID,'long_name',longnamstd) ; call erreur(status,.TRUE.,"get_att3")
status = NF90_GET_ATT(fidA,var_stdin_ID,'comment',comm) ; call erreur(status,.TRUE.,"get_att4")
status = NF90_GET_ATT(fidA,NF90_GLOBAL,'description',desc)  ; call erreur(status,.TRUE.,"get_att_glo1")
status = NF90_GET_ATT(fidA,NF90_GLOBAL,'history',his)  ; call erreur(status,.TRUE.,"get_att_glo2")

status = NF90_GET_VAR(fidA,z_ID,z); call erreur(status,.TRUE.,"getvar_z")
status = NF90_GET_VAR(fidA,z_bnds_ID,z_bnds); call erreur(status,.TRUE.,"getvar_z_bnds")
status = NF90_GET_VAR(fidA,y_ID,y); call erreur(status,.TRUE.,"getvar_y")
status = NF90_GET_VAR(fidA,y_bnds_ID,y_bnds); call erreur(status,.TRUE.,"getvar_y_bnds")
status = NF90_GET_VAR(fidA,x_ID,x); call erreur(status,.TRUE.,"getvar_x")
status = NF90_GET_VAR(fidA,x_bnds_ID,x_bnds); call erreur(status,.TRUE.,"getvar_x_bnds")
 
!---------------------------------------
! Writing new netcdf file :
 
write(*,*) 'Creating ', TRIM(file_out)

status = NF90_CREATE(TRIM(file_out),NF90_NOCLOBBER,fidM); call erreur(status,.TRUE.,'create')
 
status = NF90_DEF_DIM(fidM,"x",mx,dimID_x); call erreur(status,.TRUE.,"def_dimID_x")
status = NF90_DEF_DIM(fidM,"y",my,dimID_y); call erreur(status,.TRUE.,"def_dimID_y")
status = NF90_DEF_DIM(fidM,"z",mz,dimID_z); call erreur(status,.TRUE.,"def_dimID_z")
status = NF90_DEF_DIM(fidM,"bnds",mbnds,dimID_bnds); call erreur(status,.TRUE.,"def_dimID_bnds")
  
status = NF90_DEF_VAR(fidM,TRIM(varnam),NF90_FLOAT,(/dimID_x,dimID_y,dimID_z/),varout_ID); call erreur(status,.TRUE.,"def_varout_ID")
status = NF90_DEF_VAR(fidM,TRIM(varnam)//'_uncert',NF90_FLOAT,(/dimID_x,dimID_y,dimID_z/),var_stdout_ID); call erreur(status,.TRUE.,"def_var_stdout_ID")
status = NF90_DEF_VAR(fidM,"x",NF90_DOUBLE,(/dimID_x/),x_ID); call erreur(status,.TRUE.,"def_var_x_ID")
status = NF90_DEF_VAR(fidM,"y",NF90_DOUBLE,(/dimID_y/),y_ID); call erreur(status,.TRUE.,"def_var_y_ID")
status = NF90_DEF_VAR(fidM,"z",NF90_DOUBLE,(/dimID_z/),z_ID); call erreur(status,.TRUE.,"def_var_z_ID")
status = NF90_DEF_VAR(fidM,"x_bnds",NF90_DOUBLE,(/dimID_bnds,dimID_x/),x_bnds_ID); call erreur(status,.TRUE.,"def_var_x_bnds_ID")
status = NF90_DEF_VAR(fidM,"y_bnds",NF90_DOUBLE,(/dimID_bnds,dimID_y/),y_bnds_ID); call erreur(status,.TRUE.,"def_var_y_bnds_ID")
status = NF90_DEF_VAR(fidM,"z_bnds",NF90_DOUBLE,(/dimID_bnds,dimID_z/),z_bnds_ID); call erreur(status,.TRUE.,"def_var_z_bnds_ID")
 
status = NF90_PUT_ATT(fidM,varout_ID,"units",TRIM(uni)); call erreur(status,.TRUE.,"put_att_varout_ID")
status = NF90_PUT_ATT(fidM,varout_ID,"long_name",TRIM(longnam)); call erreur(status,.TRUE.,"put_att_varout_ID")
status = NF90_PUT_ATT(fidM,varout_ID,"_FillValue",miss); call erreur(status,.TRUE.,"put_att_varout_ID")
status = NF90_PUT_ATT(fidM,var_stdout_ID,"units",TRIM(uni)); call erreur(status,.TRUE.,"put_att_var_stdout_ID")
status = NF90_PUT_ATT(fidM,var_stdout_ID,"long_name",TRIM(longnamstd)); call erreur(status,.TRUE.,"put_att_var_stdout_ID")
status = NF90_PUT_ATT(fidM,var_stdout_ID,"comment",TRIM(comm)); call erreur(status,.TRUE.,"put_att_var_stdout_ID")
status = NF90_PUT_ATT(fidM,var_stdout_ID,"_FillValue",miss); call erreur(status,.TRUE.,"put_att_var_stdout_ID")
status = NF90_PUT_ATT(fidM,z_ID,"bounds","z_bnds"); call erreur(status,.TRUE.,"put_att_z_ID")
status = NF90_PUT_ATT(fidM,z_ID,"axis","Z"); call erreur(status,.TRUE.,"put_att_z_ID")
status = NF90_PUT_ATT(fidM,z_ID,"positive","up"); call erreur(status,.TRUE.,"put_att_z_ID")
status = NF90_PUT_ATT(fidM,z_ID,"long_name","height relative to sea surface (positive up)"); call erreur(status,.TRUE.,"put_att_z_ID")
status = NF90_PUT_ATT(fidM,z_ID,"standard_name","height"); call erreur(status,.TRUE.,"put_att_z_ID")
status = NF90_PUT_ATT(fidM,z_ID,"units","m"); call erreur(status,.TRUE.,"put_att_z_ID")
status = NF90_PUT_ATT(fidM,y_ID,"bounds","y_bnds"); call erreur(status,.TRUE.,"put_att_y_ID")
status = NF90_PUT_ATT(fidM,y_ID,"axis","Y"); call erreur(status,.TRUE.,"put_att_y_ID")
status = NF90_PUT_ATT(fidM,y_ID,"long_name","y coordinate of projection"); call erreur(status,.TRUE.,"put_att_y_ID")
status = NF90_PUT_ATT(fidM,y_ID,"standard_name","projection_y_coordinate"); call erreur(status,.TRUE.,"put_att_y_ID")
status = NF90_PUT_ATT(fidM,y_ID,"units","m"); call erreur(status,.TRUE.,"put_att_y_ID")
status = NF90_PUT_ATT(fidM,x_ID,"bounds","x_bnds"); call erreur(status,.TRUE.,"put_att_x_ID")
status = NF90_PUT_ATT(fidM,x_ID,"axis","X"); call erreur(status,.TRUE.,"put_att_x_ID")
status = NF90_PUT_ATT(fidM,x_ID,"long_name","x coordinate of projection"); call erreur(status,.TRUE.,"put_att_x_ID")
status = NF90_PUT_ATT(fidM,x_ID,"standard_name","projection_x_coordinate"); call erreur(status,.TRUE.,"put_att_x_ID")
status = NF90_PUT_ATT(fidM,x_ID,"units","m"); call erreur(status,.TRUE.,"put_att_x_ID")

status = NF90_PUT_ATT(fidM,NF90_GLOBAL,"project","ISMIP7"); call erreur(status,.TRUE.,"att_GLO1")
status = NF90_PUT_ATT(fidM,NF90_GLOBAL,"history",TRIM(his)//' then extrapolate_remaining_vertically_special.f90'); call erreur(status,.TRUE.,"put_att_GLOBAL1_ID")
status = NF90_PUT_ATT(fidM,NF90_GLOBAL,"description",TRIM(desc)); call erreur(status,.TRUE.,"put_att_GLOBAL2_ID")
 
status = NF90_ENDDEF(fidM); call erreur(status,.TRUE.,"fin_definition") 
 
status = NF90_PUT_VAR(fidM,x_ID,x); call erreur(status,.TRUE.,"var_x_ID")
status = NF90_PUT_VAR(fidM,y_ID,y); call erreur(status,.TRUE.,"var_y_ID")
status = NF90_PUT_VAR(fidM,z_ID,z); call erreur(status,.TRUE.,"var_z_ID")
status = NF90_PUT_VAR(fidM,x_bnds_ID,x_bnds); call erreur(status,.TRUE.,"var_x_bnds_ID")
status = NF90_PUT_VAR(fidM,y_bnds_ID,y_bnds); call erreur(status,.TRUE.,"var_y_bnds_ID")
status = NF90_PUT_VAR(fidM,z_bnds_ID,z_bnds); call erreur(status,.TRUE.,"var_z_bnds_ID")

!----------------------------------------------------------------------------------------
! Extrapolating vertically from kz=2 :
!
  status = NF90_GET_VAR(fidA,varin_ID,varin,start=(/1,1,1/),count=(/mx,my,mz/))
  call erreur(status,.TRUE.,"getvarin")
  status = NF90_GET_VAR(fidA,var_stdin_ID,var_stdin,start=(/1,1,1/),count=(/mx,my,mz/))
  call erreur(status,.TRUE.,"getvar_stdin")

  do ki=1,mx
  do kj=1,my
    if ( abs(varin(ki,kj,1)) .gt. 1.e3 .and. abs(varin(ki,kj,2)) .lt. 1.e3 ) then
       varin(ki,kj,1) = varin(ki,kj,2)
       var_stdin(ki,kj,1) = var_stdin(ki,kj,2)
    endif
  enddo
  enddo

  DO kz=3,mz
    do ki=1,mx
    do kj=1,my
      if ( abs(varin(ki,kj,kz)) .gt. 1.e3 .and. abs(varin(ki,kj,kz-1)) .lt. 1.e3 ) then
         varin(ki,kj,kz) = varin(ki,kj,kz-1)
         var_stdin(ki,kj,kz) = var_stdin(ki,kj,kz-1)
      endif
    enddo
    enddo
  ENDDO

  status = NF90_PUT_VAR(fidM,varout_ID,varin,start=(/1,1,1/),count=(/mx,my,mz/))
  call erreur(status,.TRUE.,"putvarout")
  status = NF90_PUT_VAR(fidM,var_stdout_ID,var_stdin,start=(/1,1,1/),count=(/mx,my,mz/))
  call erreur(status,.TRUE.,"putvar_stdout")

!----------------------------------------------------------------------------------------

status = NF90_CLOSE(fidA); call erreur(status,.TRUE.,"close_file")
status = NF90_CLOSE(fidM); call erreur(status,.TRUE.,"final")

end program modif



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
