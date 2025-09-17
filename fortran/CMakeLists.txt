cmake_minimum_required(VERSION 3.18)

project(i7aof_extrapolation LANGUAGES Fortran)

# Option to build shared libs later if refactored; currently just executables
option(BUILD_I7AOF_EXTRAP "Build ISMIP7 Antarctic ocean forcing extrapolation tools" ON)
if(NOT BUILD_I7AOF_EXTRAP)
    message(STATUS "Extrapolation tools disabled (BUILD_I7AOF_EXTRAP=OFF)")
    return()
endif()

# Allow user to specify NetCDF Fortran root
# Tries pkg-config, then nf-config, then CMake's find_package

find_package(PkgConfig QUIET)

set(NC_HINTS "${NetCDF_ROOT}" "$ENV{NetCDF_ROOT}" "$ENV{NETCDF_DIR}" "$ENV{NETCDF_ROOT}" )
list(REMOVE_DUPLICATES NC_HINTS)

set(NETCDF_FORTRAN_FOUND FALSE)

if(PkgConfig_FOUND)
    pkg_check_modules(NETCDFF QUIET netcdf-fortran)
    if(NETCDFF_FOUND)
        set(NETCDF_FORTRAN_FOUND TRUE)
        set(NETCDF_Fortran_INCLUDE_DIRS ${NETCDFF_INCLUDE_DIRS})
        set(NETCDF_Fortran_LIBRARIES ${NETCDFF_LIBRARIES})
    endif()
endif()

if(NOT NETCDF_FORTRAN_FOUND)
    # Try nf-config (provided by many netcdf-fortran installs including conda)
    find_program(NF_CONFIG nf-config HINTS ${NC_HINTS})
    if(NF_CONFIG)
        execute_process(COMMAND ${NF_CONFIG} --fflags OUTPUT_VARIABLE NF_FFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${NF_CONFIG} --flibs  OUTPUT_VARIABLE NF_FLIBS  OUTPUT_STRIP_TRAILING_WHITESPACE)
        separate_arguments(NF_FFLAGS)
        separate_arguments(NF_FLIBS)
        add_library(netcdf_fortran_interface INTERFACE)
        target_compile_options(netcdf_fortran_interface INTERFACE ${NF_FFLAGS})
        target_link_libraries(netcdf_fortran_interface INTERFACE ${NF_FLIBS})
        set(NETCDF_FORTRAN_FOUND TRUE)
        set(NETCDF_Fortran_LIB_TARGET netcdf_fortran_interface)
    endif()
endif()

if(NOT NETCDF_FORTRAN_FOUND)
    # Final attempt: use CMake's FindNetCDF if available
    find_package(NetCDF COMPONENTS Fortran)
    if(TARGET NetCDF::NetCDF_Fortran)
        set(NETCDF_FORTRAN_FOUND TRUE)
        set(NETCDF_Fortran_LIB_TARGET NetCDF::NetCDF_Fortran)
    endif()
endif()

if(NOT NETCDF_FORTRAN_FOUND)
    message(FATAL_ERROR "Could not locate NetCDF Fortran library. Provide pkg-config file, nf-config, or CMake package.")
endif()

# Sources
set(HORIZONTAL_SRC extrapolate_everywhere_horizontally.f90)
set(VERTICAL_SRC   extrapolate_remaining_vertically.f90)

add_executable(i7aof_extrap_horizontal ${HORIZONTAL_SRC})
add_executable(i7aof_extrap_vertical   ${VERTICAL_SRC})

# If we only have include dirs + libs variables (pkg-config path)
if(NETCDF_Fortran_INCLUDE_DIRS)
    target_include_directories(i7aof_extrap_horizontal PRIVATE ${NETCDF_Fortran_INCLUDE_DIRS})
    target_include_directories(i7aof_extrap_vertical   PRIVATE ${NETCDF_Fortran_INCLUDE_DIRS})
endif()

if(NETCDF_Fortran_LIBRARIES)
    target_link_libraries(i7aof_extrap_horizontal PRIVATE ${NETCDF_Fortran_LIBRARIES})
    target_link_libraries(i7aof_extrap_vertical   PRIVATE ${NETCDF_Fortran_LIBRARIES})
elseif(NETCDF_Fortran_LIB_TARGET)
    target_link_libraries(i7aof_extrap_horizontal PRIVATE ${NETCDF_Fortran_LIB_TARGET})
    target_link_libraries(i7aof_extrap_vertical   PRIVATE ${NETCDF_Fortran_LIB_TARGET})
endif()

# Basic compile flags (user can override with CMAKE_Fortran_FLAGS)
if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    target_compile_options(i7aof_extrap_horizontal PRIVATE -O2 -fimplicit-none -Wall)
    target_compile_options(i7aof_extrap_vertical   PRIVATE -O2 -fimplicit-none -Wall)
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    target_compile_options(i7aof_extrap_horizontal PRIVATE -O2 -warn all)
    target_compile_options(i7aof_extrap_vertical   PRIVATE -O2 -warn all)
endif()

install(TARGETS i7aof_extrap_horizontal i7aof_extrap_vertical
        RUNTIME DESTINATION bin)

message(STATUS "Configured i7aof_extrap_horizontal and i7aof_extrap_vertical")
