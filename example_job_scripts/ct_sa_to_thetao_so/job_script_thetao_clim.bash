#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --job-name=ismip7_thetao_so_clim
#SBATCH --output=ismip7_thetao_so_clim.o%j
#SBATCH --error=ismip7_thetao_so_clim.e%j

export OMP_NUM_THREADS=1

source ~/chrysalis/miniforge3/etc/profile.d/conda.sh
conda activate ismip7_dev

set -e

export HDF5_USE_FILE_LOCKING=FALSE

model="CESM2-WACCM"
clim_name="zhou_annual_30_sep"
scenario="ssp585"
inputdir="/lcrc/group/e3sm/ac.xylar/ismip7"
workdir="/lcrc/group/e3sm/ac.xylar/ismip7/full_workflow"

echo ismip7-antarctic-clim-ct-sa-to-thetao-so \
    --clim $clim_name \
    --workdir $workdir

ismip7-antarctic-clim-ct-sa-to-thetao-so  \
    --clim $clim_name \
    --workdir $workdir

echo "All done!"
