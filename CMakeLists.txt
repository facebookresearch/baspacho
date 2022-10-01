#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

cmake_minimum_required(VERSION 3.20)

# set the project name
project(BaSpaCho CXX)
set(CMAKE_CXX_STANDARD 17)
set(BASPACHO_CXX_FLAGS -Wall -Wextra -pedantic -Werror)

message("* Build type: " ${CMAKE_BUILD_TYPE})
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/BundleStaticLibrary.cmake")

# general settings
set(BASPACHO_BUILD_TESTS ON CACHE BOOL "If on (default), tests are build")
set(BASPACHO_BUILD_EXAMPLES ON CACHE BOOL "If on (default), exampels/benchmarks are build")

# some fancy colors for cmake messages
if(NOT WIN32)
string(ASCII 27 Esc)
set(ColourReset "${Esc}[m")
set(Red         "${Esc}[31m")
set(Green       "${Esc}[32m")
set(Yellow      "${Esc}[33m")
set(Blue        "${Esc}[34m")
set(Magenta     "${Esc}[35m")
set(Cyan        "${Esc}[36m")
set(White       "${Esc}[37m")
endif()


# CUDA
set(BASPACHO_USE_CUBLAS ON CACHE BOOL "If on, CUDA support is enabled")
set(BASPACHO_CUDA_ARCHS "detect" CACHE STRING "List of cuda architecture (eg '60;62;75'), or 'detect', or 'torch'")
if(BASPACHO_USE_CUBLAS)
  message("${Cyan}==============================[ CUDA ]==================================${ColourReset}")
  enable_language(CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  if(BASPACHO_CUDA_ARCHS STREQUAL "detect")
    message("* Detecting CUDA architecture...")
    set(CUDA_DETECT_BIN "${CMAKE_BINARY_DIR}/detect_cuda")
    set(CUDA_DETECT_SRC "${CMAKE_CURRENT_SOURCE_DIR}/cmake/detect_cuda.cu")

    message("${Yellow}  ${CMAKE_CUDA_COMPILER} ${CUDA_DETECT_SRC} -o ${CUDA_DETECT_BIN}${ColourReset}")
    execute_process(COMMAND "${CMAKE_CUDA_COMPILER}" "${CUDA_DETECT_SRC}" -o "${CUDA_DETECT_BIN}"
                    OUTPUT_VARIABLE NVCC_OUT
                    ERROR_VARIABLE NVCC_ERR)
    if(NOT NVCC_OUT STREQUAL "")
      message("${NVCC_OUT}")
    endif()
    if(NOT NVCC_ERR STREQUAL "")
      message("${Red}${NVCC_ERR}${ColourReset}")
    endif()
    execute_process(COMMAND "${CUDA_DETECT_BIN}"
                    RESULT_VARIABLE CUDA_RETURN_CODE
                    OUTPUT_VARIABLE RESULT)

    if(CUDA_RETURN_CODE EQUAL 0)
      message("* CUDA architectures: '${RESULT}' (autodetected installed GPU)")
      set(BASPACHO_CUDA_ARCHITECTURES "${RESULT}")
    else()
      message(FATAL_ERROR "CUDA architecture detection failed: ${RESULT}")
    endif()
	elseif(BASPACHO_CUDA_ARCHS STREQUAL "torch")
    message("* Getting Torch CUDA architectures...")
    set(CUDA_GET_TORCH_ARCHS_BIN "${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_torch_cuda_archs.py")
    message("${Yellow}  python ${CUDA_GET_TORCH_ARCHS_BIN}${ColourReset}")
		execute_process(COMMAND "python" "${CUDA_GET_TORCH_ARCHS_BIN}"
                    RESULT_VARIABLE TORCH_RETURN_CODE
                    OUTPUT_VARIABLE RESULT
                    ERROR_VARIABLE TORCH_ERROR)

    if(NOT TORCH_ERROR STREQUAL "")
      message("${Red}${TORCH_ERROR}${ColourReset}")
    endif()

    if(TORCH_RETURN_CODE EQUAL 0)
      message("* CUDA architectures: '${RESULT}' (from torch, >=60)")
		  if(RESULT STREQUAL "")
				message(FATAL_ERROR "Torch architectures returned an empty result.")
			endif()
      set(BASPACHO_CUDA_ARCHITECTURES "${RESULT}")
    else()
      message(FATAL_ERROR "Torch CUDA architecture detection failed: ${RESULT}")
    endif()
  else()
    # https://cmake.org/cmake/help/git-stage/prop_tgt/CUDA_ARCHITECTURES.html
    message("* CUDA architectures: '${BASPACHO_CUDA_ARCHS}' (manually set)")
    set(BASPACHO_CUDA_ARCHITECTURES "${BASPACHO_CUDA_ARCHS}")
  endif()

  list(APPEND CUDA_NVCC_FLAGS
       -Xcompiler -Wall,-Wextra,-Werror,-Wno-unknown-pragmas
       --expt-relaxed-constexpr)
  message("* CUDA Includes : " ${CUDAToolkit_INCLUDE_DIRS})
  message("* CUDA Libraries: " ${CUDAToolkit_LIBRARY_DIR})
  include_directories(${CUDAToolkit_INCLUDE_DIRS})
  add_compile_definitions(BASPACHO_USE_CUBLAS)
endif()


# BLAS. a few possibilities are:
# * ATLAS
# * OpenBLAS
# * Intel10_64{i}lp{_seq}
#   (if 'i' indices are 64bits, if '_seq' sequential ie single thread)
set(BASPACHO_USE_BLAS ON CACHE BOOL "If on, BLAS support is enabled")
if(BASPACHO_USE_BLAS)
  message("${Cyan}==============================[ BLAS ]==================================${ColourReset}")
  set(BLA_VENDOR OpenBLAS CACHE STRING
      "BLAS Library to be searched, possibilities are: ATLAS, OpenBLAS, Intel10_64ilp_seq, etc")
  set(BLA_STATIC OFF CACHE STRING
      "Link statically to BLAS Library")
  message("* BLA_VENDOR = " ${BLA_VENDOR})
  message("* BLA_STATIC = " ${BLA_STATIC})
  find_package(BLAS REQUIRED)
  message("* BLAS linkerFlags = " ${BLAS_LINKER_FLAGS})
  message("* BLAS Libraries = " ${BLAS_LIBRARIES})
  if(BLA_VENDOR MATCHES "Intel10_*")
    message("* BLAS appears to be MKL, MKL_ROOT should be set")
    message("* Adding includes: " $ENV{MKLROOT}/include)
    include_directories($ENV{MKLROOT}/include) # if MKL
    add_compile_definitions(BASPACHO_USE_MKL)
  endif()
  add_compile_definitions(BASPACHO_USE_BLAS)
endif()


# amd, if using amd from SuiteSparse
message("${Cyan}===============[ minimum degree reordering algorithm ]==================${ColourReset}")
set(BASPACHO_USE_SUITESPARSE_AMD OFF CACHE BOOL "If on, will use AMD algorithm in SuiteSparse (must be installed)")
if(BASPACHO_USE_SUITESPARSE_AMD)
  set(HAVE_SUITESPARSE_AMD 1)
  find_path(SUITESPARSE_AMD_INCLUDE_DIR NAMES amd.h
            PATHS $ENV{AMDDIR} ${INCLUDE_INSTALL_DIR}
            PATH_SUFFIXES suitesparse ufsparse
            REQUIRED)
  find_library(SUITESPARSE_AMD_LIBRARIES amd
              PATHS $ENV{AMDDIR} ${LIB_INSTALL_DIR}
              REQUIRED)
  include_directories(${SUITESPARSE_AMD_INCLUDE_DIR})
  message("* Using Approximate Minimum Degree (AMD) algorithm from SuiteSparse")
  message("* Amd Source = " ${SUITESPARSE_AMD_INCLUDE_DIR})
  message("* Amd Libraries = " ${SUITESPARSE_AMD_LIBRARIES})
  add_compile_definitions(BASPACHO_USE_SUITESPARSE_AMD)
else()
  set(HAVE_SUITESPARSE_AMD 0)
  message("* Using Approximate Minimum Degree (AMD) algorithm from Eigen")
endif()


# cholmod (only needed for benchmarking comparison)
if(BASPACHO_BUILD_EXAMPLES)
  message("${Cyan}============================[ CHOLMOD ]=================================${ColourReset}")
  find_path(CHOLMOD_INCLUDE_DIR NAMES cholmod.h
            PATHS ${INCLUDE_INSTALL_DIR}
            PATH_SUFFIXES suitesparse ufsparse)
  find_library(CHOLMOD_LIBRARIES cholmod
              PATHS ${LIB_INSTALL_DIR})
  if((CHOLMOD_INCLUDE_DIR STREQUAL "CHOLMOD_INCLUDE_DIR-NOTFOUND")
    OR (CHOLMOD_LIBRARIES STREQUAL "CHOLMOD_LIBRARIES-NOTFOUND"))
    set(HAVE_CHOLMOD 0)
    message("* Cholmod not found! Benchmarking tools will not support it")
  else()
    set(HAVE_CHOLMOD 1)
    message("* Cholmod Source = " ${CHOLMOD_INCLUDE_DIR})
    message("* Cholmod Libraries = " ${CHOLMOD_LIBRARIES})
    add_compile_definitions(BASPACHO_HAVE_CHOLMOD)
    include_directories(${CHOLMOD_INCLUDE_DIR})
  endif()
endif()


# use fetchcontent to download dependencies
include(FetchContent)
cmake_policy(SET CMP0077 NEW)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
	cmake_policy(SET CMP0135 NEW)
endif()

# dispenso for multithreading
# currently in facebookincubator, soon to be opensourced
message("${Cyan}============================[ dispenso ]================================${ColourReset}")
FetchContent_Declare(
  dispenso
  URL https://github.com/facebookincubator/dispenso/archive/552738b62a04f5d7b51dfcef822ae2d89d154881.zip
  )
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
FetchContent_MakeAvailable(dispenso)
unset(CMAKE_POSITION_INDEPENDENT_CODE)
include_directories(${dispenso_SOURCE_DIR} ${dispenso_SOURCE_DIR}/dispenso/third-party/moodycamel)
message("* Dispenso Source = " ${dispenso_SOURCE_DIR})
message("* Dispenso Library Dir = " ${dispenso_BINARY_DIR})


# gtest
if(BASPACHO_BUILD_TESTS)
  message("${Cyan}=============================[ GTest ]==================================${ColourReset}")
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
    )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  message("* GTest Source = " ${gtest_SOURCE_DIR})
  message("* GTest Library Dir = " ${gtest_BINARY_DIR})
  message("* GMock Source = " ${gmock_SOURCE_DIR})
  message("* GMock Library Dir = " ${gmock_BINARY_DIR})
  include(GoogleTest)
endif()


# eigen
message("${Cyan}=============================[ Eigen ]==================================${ColourReset}")
FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
	)
if(NOT eigen_POPULATED)
  FetchContent_Populate(eigen)
endif()
message("* Eigen Source = " ${eigen_SOURCE_DIR})
message("* Eigen Library Dir = " ${eigen_BINARY_DIR})
include_directories(${eigen_SOURCE_DIR})


# Sophus (only needed for BA problem in examples/benchmarks)
if(BASPACHO_BUILD_EXAMPLES)
  message("${Cyan}=============================[ Sophus ]=================================${ColourReset}")
  FetchContent_Declare(
    sophus
    URL https://github.com/strasdat/Sophus/archive/8e1941677ef214aab0397608f42b18fbe1a19885.zip
  )
  if(NOT sophus_POPULATED)
    FetchContent_Populate(sophus)
  endif()
  message("* Sophus Source = " ${sophus_SOURCE_DIR})
  message("* Sophus Library Dir = " ${sophus_BINARY_DIR})
  include_directories(${sophus_SOURCE_DIR})
  add_compile_definitions(SOPHUS_USE_BASIC_LOGGING=1)
endif()


# add root to include path
include_directories("${PROJECT_SOURCE_DIR}")


# basic lib name depends on shared/static, so that
# `BaSpaCho` is the bundled library if static
if(BUILD_SHARED_LIBS)
    set(BASPACHO_LIBRARY BaSpaCho)
else()
    set(BASPACHO_LIBRARY BaSpaCho_static)
endif()


# unit tests
if(BASPACHO_BUILD_TESTS)
  enable_testing()

  function(add_baspacho_test TARGET)
    add_executable(
      "${TARGET}"
      "${ARGN}"
    )
    target_link_libraries(
      "${TARGET}" ${BASPACHO_LIBRARY} testing gtest_main gmock
    )
    gtest_discover_tests("${TARGET}")
  endfunction()
endif()


# add subfolders
add_subdirectory(baspacho)
