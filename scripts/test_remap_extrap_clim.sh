#!/bin/bash -l

export OMP_NUM_THREADS=1

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ismip7_dev

export HDF5_USE_FILE_LOCKING=FALSE

ismip_base = /lcrc/group/e3sm/ac.xylar/ismip7
work_base = ${ismip_base}/test_remap_clim

for clim in zhou_annual_30_sep zhou_summer_30_sep zhou_2000_annual_30_sep
do

    ismip7-antarctic-remap-clim \
        -n ${clim} \
        -i ${ismip_base} \
        -w ${work_base}

    ismip7-antarctic-extrap-clim \
        -n ${clim} \
        -w ${work_base}
done
