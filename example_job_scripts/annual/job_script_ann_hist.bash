#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=ismip7_ann_hist
#SBATCH --output=ismip7_ann_hist.o%j
#SBATCH --error=ismip7_ann_hist.e%j

export OMP_NUM_THREADS=1

source ~/chrysalis/miniforge3/etc/profile.d/conda.sh
conda activate ismip7_dev

set -e

export HDF5_USE_FILE_LOCKING=FALSE

model="CESM2-WACCM"
clim_name="zhou_annual_30_sep"
scenario="historical"
inputdir="/lcrc/group/e3sm/ac.xylar/ismip7/CMIP6_test_protocol"
workdir="/lcrc/group/e3sm/ac.xylar/ismip7/full_workflow"


echo ismip7-antarctic-cmip-annual-averages \
    --model $model \
    --scenario $scenario \
    --clim $clim_name \
    --workdir $workdir

ismip7-antarctic-cmip-annual-averages \
    --model $model \
    --scenario $scenario \
    --clim $clim_name \
    --workdir $workdir

echo "All done!"
