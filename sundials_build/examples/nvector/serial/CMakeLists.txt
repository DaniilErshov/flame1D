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
# CMakeLists.txt for  C serial examples.
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
project(_C_serial_examples C)

# Enable testing
include(CTest)

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
  sundials_nvecserial ${SUNDIALS_LIBRARY_DIR}
  DOC " library")

find_library(SUNDIALS_NVEC_LIB
  sundials_nvecserial ${SUNDIALS_LIBRARY_DIR}
  DOC "NVECTOR_SERIAL library")

find_library(SUNDIALS_MANYVEC_LIB
  sundials_nvecmanyvector ${SUNDIALS_LIBRARY_DIR}
  DOC "NVECTOR_MANYVECTOR library")

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
  ${SUNDIALS_MANYVEC_LIB}
  ${SUNDIALS_EXTRA_LIBS})

# ------------------------------------------------------------------------------

# Set the names of the examples to be built and their dependencies
set(examples  test_nvector_serial)
set(examples_dependencies  test_nvector.c)
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

# ------------------------------------------------------------------------------

# Set LAPACK libraries
set(LAPACK_LIBRARIES
  
  CACHE STRING "Lapack libraries")

# Set the names of the examples to be built and their dependencies
set(examples_bl )
set(examples_dependencies_bl )
if(examples_bl)
  list(REMOVE_DUPLICATES examples_bl)
endif()

if(LAPACK_LIBRARIES AND examples_bl)

  # Find the SUNDIALS linear solver libraries
  find_library(SUNDIALS_SUNLINSOLLAPACKBAND_LIB
    sundials_sunlinsollapackband ${SUNDIALS_LIBRARY_DIR}
    DOC "SUNDIALS banded LAPACK linear solver library")

  find_library(SUNDIALS_SUNLINSOLLAPACKDENSE_LIB
    sundials_sunlinsollapackdense ${SUNDIALS_LIBRARY_DIR}
    DOC "SUNDIALS dense LAPACK linear solver library")

  # Create targets for each example
  foreach(example ${examples_bl})

    # example source files
    add_executable(${example} ${example}.c ${examples_dependencies_bl})

    # directories to include
    target_include_directories(${example} PRIVATE ${SUNDIALS_INCLUDE_DIR})

    # libraries to link against
    target_link_libraries(${example} ${SUNDIALS_LIBRARIES})
    target_link_libraries(${example} ${SUNDIALS_SUNLINSOLLAPACKBAND_LIB})
    target_link_libraries(${example} ${SUNDIALS_SUNLINSOLLAPACKDENSE_LIB})
    target_link_libraries(${example} ${LAPACK_LIBRARIES})

    # add the example to ctest
    add_test(NAME ${example} COMMAND ${example})

  endforeach()

endif()

# ------------------------------------------------------------------------------

# KLU include directory and libraries
set(KLU_INCLUDE_DIR
  
  CACHE PATH "Location of KLU header files")

set(KLU_LIBRARIES
  
  CACHE STRING "KLU libraries")

# Set the names of the examples to be built and their dependencies
set(examples_klu )
set(examples_dependencies_klu )
if(examples_klu)
  list(REMOVE_DUPLICATES examples_klu)
endif()

if(KLU_LIBRARIES AND examples_klu)

  # Find the SUNDIALS linear solver libraries
  find_library(SUNDIALS_SUNLINSOLKLU_LIB
    sundials_sunlinsolklu ${SUNDIALS_LIBRARY_DIR}
    DOC "SUNDIALS KLU linear solver library")

  # Create targets for each example
  foreach(example ${examples_klu})

    # example source files
    add_executable(${example} ${example}.c ${examples_dependencies_klu})

    # directories to include
    target_include_directories(${example} PRIVATE ${SUNDIALS_INCLUDE_DIR})
    target_include_directories(${example} PRIVATE ${KLU_INCLUDE_DIR})

    # libraries to link against
    target_link_libraries(${example} ${SUNDIALS_LIBRARIES})
    target_link_libraries(${example} ${SUNDIALS_SUNLINSOLKLU_LIB})
    target_link_libraries(${example} ${KLU_LIBRARIES})

    # add the example to ctest
    add_test(NAME ${example} COMMAND ${example})

  endforeach()

endif()

# ------------------------------------------------------------------------------

# SuperLU_MT include directory, libraries, and threading type
set(SUPERLUMT_INCLUDE_DIR
  
  CACHE PATH "Location of SuperLU_MT KLU header files")

set(SUPERLUMT_LIBRARIES
   
  CACHE STRING "SuperLU_MT libraries")

set(SUPERLUMT_THREAD_TYPE
  
  CACHE STRING "SuperLU_MT Threading")

# Set the names of the examples to be built and their dependencies
set(examples_slumt )
set(examples_dependencies_slumt )
if(examples_slumt)
  list(REMOVE_DUPLICATES examples_slumt)
endif()

if(SUPERLUMT_LIBRARIES AND examples_slumt)

  # Find the SUNDIALS linear solver libraries
  find_library(SUNDIALS_SUNLINSOLSLUMT_LIB
    sundials_sunlinsolsuperlumt ${SUNDIALS_LIBRARY_DIR}
    DOC "SUNDIALS SuperLU_MT linear solver library")

  if(SUPERLUMT_THREAD_TYPE STREQUAL "OPENMP")
    find_package(OpenMP REQUIRED)
    # Update C compiler and linker flags
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_C_FLAGS}")
  else(SUPERLUMT_THREAD_TYPE STREQUAL "PTHREAD")
    find_package(Threads REQUIRED)
  endif()

  # Create targets for each example
  foreach(example ${examples_slumt})

    # example source files
    add_executable(${example} ${example}.c ${examples_dependencies_slumt})

    # directories to include
    target_include_directories(${example} PRIVATE ${SUNDIALS_INCLUDE_DIR})
    target_include_directories(${example} PRIVATE ${SUPERLUMT_INCLUDE_DIR})

    # libraries to link against
    target_link_libraries(${example} ${SUNDIALS_LIBRARIES})
    target_link_libraries(${example} ${SUNDIALS_SUNLINSOLSLUMT_LIB})
    target_link_libraries(${example} ${SUPERLUMT_LIBRARIES})

    if(SUPERLUMT_THREAD_TYPE STREQUAL "PTHREAD")
      target_link_libraries(${example} Threads::Threads)
    endif()

    # add the example to ctest
    add_test(NAME ${example} COMMAND ${example})

  endforeach()

endif()
