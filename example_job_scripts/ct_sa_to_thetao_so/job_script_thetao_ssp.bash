#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=ismip7_thetao_so_ssp
#SBATCH --output=ismip7_thetao_so_ssp.o%j
#SBATCH --error=ismip7_thetao_so_ssp.e%j

export OMP_NUM_THREADS=1

source ~/chrysalis/miniforge3/etc/profile.d/conda.sh
conda activate ismip7_dev

set -e

export HDF5_USE_FILE_LOCKING=FALSE

model="CESM2-WACCM"
clim_name="zhou_annual_30_sep"
scenario="ssp585"
inputdir="/lcrc/group/e3sm/ac.xylar/ismip7/CMIP6_test_protocol"
workdir="/lcrc/group/e3sm/ac.xylar/ismip7/full_workflow"


echo ismip7-antarctic-cmip-annual-ct-sa-to-thetao-so \
    --model $model \
    --scenario $scenario \
    --clim $clim_name \
    --workdir $workdir

ismip7-antarctic-cmip-annual-ct-sa-to-thetao-so \
    --model $model \
    --scenario $scenario \
    --clim $clim_name \
    --workdir $workdir

echo "All done!"
