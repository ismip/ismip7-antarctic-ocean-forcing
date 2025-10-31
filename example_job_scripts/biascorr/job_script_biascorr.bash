#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=ismip7_biascorr
#SBATCH --output=ismip7_biascorr.o%j
#SBATCH --error=ismip7_biascorr.e%j

export OMP_NUM_THREADS=1

source ~/chrysalis/miniforge3/etc/profile.d/conda.sh
conda activate ismip7_dev

set -e

export HDF5_USE_FILE_LOCKING=FALSE

model="CESM2-WACCM"
clim_name="zhou_annual_30_sep"
inputdir="/lcrc/group/e3sm/ac.xylar/ismip7/CMIP6_test_protocol"
workdir="/lcrc/group/e3sm/ac.xylar/ismip7/full_workflow"

for scenario in "historical" "ssp585"; do
    echo ismip7-antarctic-bias-corr-classic \
        --model $model \
        --scenario $scenario \
        --clim $clim_name \
        --workdir $workdir

    ismip7-antarctic-bias-corr-classic \
        --model $model \
        --scenario $scenario \
        --clim $clim_name \
        --workdir $workdir
done

echo "All done!"
