#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --threads-per-core=1
#SBATCH -J extrapolate
#SBATCH -e extrapolate.e%j
#SBATCH -o extrapolate.o%j
#SBATCH --time=01:59:00

date

for PREFIX in 'Jourdain-Naughten_NEMO-MITgcm' 'Naughten_MITamu-MITwed'
do

for CW in 'cold' 'warm'
do

for VAR in 'S' 'T'
do

file="${PREFIX}_${CW}_${VAR}.nc"
echo $file

if [ $VAR == 'S' ]; then
  VARNAM="so"
else
  VARNAM="thetao"
fi
echo $VARNAM

rm -f tmp_hor.nc
sed -e "s#<file_in>#${file}#g ; s#<var_name>#${VARNAM}#g" extrapolate_everywhere_horizontally_special.f90 > tmp.f90
ifort -c $NC_INC tmp.f90
ifort -o tmp tmp.o $NC_LIB
./tmp

if [ ! -f tmp_hor.nc ]; then
  echo "~!@%^&* ERROR: tmp_hor.nc HAS NOT BEEN CREATED  >>>>>>>>>>>>>>> STOP !"
  exit
else
  date
  echo 'Horizontal extrapolation [done]'
fi

rm -f tmp tmp.o tmp.f90
sed -e "s#<file_out>#PROCESSED_OUTPUTS/${file}#g ; s#<var_name>#${VARNAM}#g" extrapolate_remaining_vertically_special.f90 > tmp.f90
ifort -c $NC_INC tmp.f90
ifort -o tmp tmp.o $NC_LIB
./tmp
rm -f tmp tmp.o tmp.f90

done

done

done

date
