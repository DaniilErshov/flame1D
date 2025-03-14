# -----------------------------------------------------------------
# Programmer(s): Radu Serban @ LLNL
#                David J. Gardner @ LLNL
# -----------------------------------------------------------------
# SUNDIALS Copyright Start
# Copyright (c) 2002-2023, Lawrence Livermore National Security
# and Southern Methodist University.
# All rights reserved.
#
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
# SUNDIALS Copyright End
# -----------------------------------------------------------------
# CMakeLists.txt for IDAS C OpenMP examples.
#
# This file is generated from a template using variables
# set at configuration time. It can be used as a template for
# other user CMakeLists configuration files.
# -----------------------------------------------------------------

# Set the minimum required cmake version
cmake_minimum_required(VERSION 3.28.3)

# Set cache variables for compilers and flags
set(CMAKE_C_COMPILER
  C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.40.33807/bin/Hostx64/x64/cl.exe
  CACHE FILEPATH "C compiler")

set(CMAKE_C_FLAGS
  "/DWIN32 /D_WINDOWS /W3"
  CACHE STRING "C compiler flags")

if("99")
  set(CMAKE_C_STANDARD "99"
    CACHE STRING "C standard")
endif()

# Specify project name and languages
project(IDAS_C_openmp_examples C)

# Enable testing
include(CTest)

# ------------------------------------------------------------------------------

# find OpenMP
find_package(OpenMP REQUIRED)

# Update C compiler and linker flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_C_FLAGS}")

# ------------------------------------------------------------------------------

# Specify the path to SUNDIALS header files
set(SUNDIALS_INCLUDE_DIR
  C:/Program Files (x86)/SUNDIALS/include
  CACHE PATH "Location of SUNDIALS header files")

# Specify the path to SUNDIALS libraries
set(SUNDIALS_LIBRARY_DIR
  C:/Program Files (x86)/SUNDIALS/lib
  CACHE PATH "Location of SUNDIALS libraries")

# Find the SUNDIALS libraries
find_library(SUNDIALS_SOLVER_LIB
  sundials_idas ${SUNDIALS_LIBRARY_DIR}
  DOC "IDAS library")

find_library(SUNDIALS_NVEC_LIB
  sundials_nvecopenmp ${SUNDIALS_LIBRARY_DIR}
  DOC "NVECTOR_OPENMP library")

# Set additional libraries
set(SUNDIALS_EXTRA_LIBS  CACHE STRING "Additional libraries")

# For SUNDIALS module examples the solver library is not needed
if(NOT SUNDIALS_SOLVER_LIB)
  set(SUNDIALS_SOLVER_LIB "")
endif()

# List of SUNDIALS libraries
set(SUNDIALS_LIBRARIES
  -L${SUNDIALS_LIBRARY_DIR}
  ${SUNDIALS_SOLVER_LIB}
  ${SUNDIALS_NVEC_LIB}
  ${SUNDIALS_EXTRA_LIBS})

# ------------------------------------------------------------------------------

# Set the names of the examples to be built and their dependencies
set(examples  idasFoodWeb_bnd_omp idasFoodWeb_kry_omp)
set(examples_dependencies )
if(examples)
  list(REMOVE_DUPLICATES examples)
endif()

# Create targets for each example
foreach(example ${examples})

  # example source files
  add_executable(${example} ${example}.c ${examples_dependencies})

  # directories to include
  target_include_directories(${example} PRIVATE ${SUNDIALS_INCLUDE_DIR})

  # libraries to link against
  target_link_libraries(${example} ${SUNDIALS_LIBRARIES})

  # add the example to ctest
  add_test(NAME ${example} COMMAND ${example})

endforeach()
