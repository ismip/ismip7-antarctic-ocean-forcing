# Processing of outputs from regional simulations

### Prerequisite

You need fortran compiler, netcdf and fortran-netcdf libraries, then export pathways for the fortran compiler, e.g.:

```bash
export FC='ifort'

export NC_INC='-I /net/nfs/tools/u20/22.3/PrgEnv/intel/linux-ubuntu20.04-zen2/netcdf-c/4.7.4-intel-2021.4.0-hivekrf7t2hvxleje5gpvqr7cvrmd3yr/include -I/net/nfs/tools/u20/22.3/PrgEnv/intel/linux-ubuntu20.04-zen2/netcdf-fortran/4.5.3-intel-2021.4.0-5umajavtlalwigosfjcw53s4ed4kkmj6/include'

export NC_LIB='-L/net/nfs/tools/u20/22.3/PrgEnv/intel/linux-ubuntu20.04-zen2/netcdf-c/4.7.4-intel-2021.4.0-hivekrf7t2hvxleje5gpvqr7cvrmd3yr/lib -lnetcdf -L/net/nfs/tools/u20/22.3/PrgEnv/intel/linux-ubuntu20.04-zen2/netcdf-fortran/4.5.3-intel-2021.4.0-5umajavtlalwigosfjcw53s4ed4kkmj6/lib -lnetcdff'
```

### Processing files

We start from the netcdf files containing model data interpolated onto the ISMIP7 8km grid, with NaNs outside of the area convered by the regional model.

These 2 scripts interpolate the cold and warm state of Jourdain et al. (2022) in the Bellingshausen-Amundsen seas and of Naughten et al. (2021) in the Weddell Sea (1pctCO2):
* extract\_Nico\_Kaitlin\_cold.f90
* extract\_Nico\_Kaitlin\_cold.f90

These 2 scripts interpolate the cold and warm state of Naughten et al. (2023) in the Bellingshausen-Amundsen seas and of Naughten et al. (2021) in the Weddell Sea (abrupt-4xCO2):
* extract\_Kaitlin\_2\_cold.f90
* extract\_Kaitlin\_2\_warm.f90

For example, do:
```bash
$FC $NC_INC extract_Nico_Kaitlin_cold.f90
$FC -o go extract_Nico_Kaitlin_cold.o $NC_LIB
./go
```

Then, after using the 4 programs above, execute the bash script:
```bash
bash extrapolate_ALL.sh
```
which makes use of extrapolate\_everywhere\_horizontally\_special.f90 and extrapolate\_remaining\_vertically\_special.f90 for all the files to create. 
