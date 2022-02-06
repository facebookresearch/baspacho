cmake_minimum_required(VERSION 3.20)

# set the project name
project(BaSpaCho CXX)
set(CMAKE_CXX_STANDARD 17)
set(BASPACHO_CXX_FLAGS -Wall -Wextra -pedantic -Werror)

message("* Build type: " ${CMAKE_BUILD_TYPE})

# CUDA
set(BASPACHO_USE_CUBLAS ON CACHE BOOL "If on, CUDA support is enabled")
if(BASPACHO_USE_CUBLAS)
  message("==============================[ CUDA ]==================================")
  enable_language(CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  # https://cmake.org/cmake/help/git-stage/prop_tgt/CUDA_ARCHITECTURES.html
  set(BASPACHO_CUDA_ARCHITECTURES "75")

  list(APPEND CUDA_NVCC_FLAGS
       # -Xcompiler -Wall,-Wextra,-Werror,-Wno-unknown-pragmas
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
message("==============================[ BLAS ]==================================")
set(BLA_VENDOR OpenBLAS CACHE STRING
    "BLAS Library to be searched, possibilities are: ATLAS, OpenBLAS, Intel10_64ilp_seq, etc")
message("* BLA_VENDOR = " ${BLA_VENDOR})
find_package(BLAS REQUIRED)
message("* BLAS linkerFlags = " ${BLAS_LINKER_FLAGS})
message("* BLAS Libraries = " ${BLAS_LIBRARIES})
if(BLA_VENDOR MATCHES "Intel10_*")
  message("* BLAS appears to be MKL, MKL_ROOT should be set")
  message("* Adding includes: " $ENV{MKLROOT}/include)
  include_directories($ENV{MKLROOT}/include) # if MKL
  add_compile_definitions(BASPACHO_USE_MKL)
endif()

# amd, if using amd from SuiteSparse
message("===============[ minimum degree reordering algorithm ]==================")
set(BASPACHO_USE_SUITESPARSE_AMD OFF CACHE BOOL "If on, will use AMD algorithm in SuiteSparse (must be installed)")
if(BASPACHO_USE_SUITESPARSE_AMD)
  set(HAVE_AMD 1)
  find_path(AMD_INCLUDE_DIR NAMES amd.h
            PATHS $ENV{AMDDIR} ${INCLUDE_INSTALL_DIR}
            PATH_SUFFIXES suitesparse ufsparse
            REQUIRED)
  find_library(AMD_LIBRARIES amd
              PATHS $ENV{AMDDIR} ${LIB_INSTALL_DIR}
              REQUIRED)
  include_directories(${AMD_INCLUDE_DIR})
  message("* Using Algebraic Minimum Degree (AMD) algorithm from SuiteSparse")
  message("* Amd Source = " ${AMD_INCLUDE_DIR})
  message("* Amd Libraries = " ${AMD_LIBRARIES})
  add_compile_definitions(BASPACHO_USE_SUITESPARSE_AMD)
else()
  set(HAVE_AMD 0)
  message("* Using Algebraic Minimum Degree (AMD) algorithm from Eigen")
endif()


# cholmod (for benchmarking comparison)
message("============================[ CHOLMOD ]=================================")
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


# use fetchcontent to download dependencies
include(FetchContent)
cmake_policy(SET CMP0077 NEW)

# dispenso for multithreading
# currently in facebookincubator, soon to be opensourced
message("============================[ dispenso ]================================")
FetchContent_Declare(
  dispenso
  GIT_REPOSITORY git@github.com:facebookincubator/dispenso.git
  GIT_TAG 22ff47d07c32303a5059bfec26304a68dff01e65
	)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
FetchContent_MakeAvailable(dispenso)
unset(CMAKE_POSITION_INDEPENDENT_CODE)
include_directories(${dispenso_SOURCE_DIR} ${dispenso_SOURCE_DIR}/dispenso/third-party/moodycamel)
message("* Dispenso Source = " ${dispenso_SOURCE_DIR})
message("* Dispenso Library Dir = " ${dispenso_BINARY_DIR})


# gtest
message("=============================[ GTest ]==================================")
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


# eigen
message("=============================[ Eigen ]==================================")
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


# sophus (for testing on BA problem)
message("=============================[ Sophus ]=================================")
FetchContent_Declare(
  sophus
  GIT_REPOSITORY git@github.com:strasdat/Sophus.git # https://github.com/
  GIT_TAG 49a7e1286910019f74fb4f0bb3e213c909f8e1b7
)
if(NOT sophus_POPULATED)
  FetchContent_Populate(sophus)
endif()
message("* Sophus Source = " ${sophus_SOURCE_DIR})
message("* Sophus Library Dir = " ${sophus_BINARY_DIR})
include_directories(${sophus_SOURCE_DIR})
add_compile_definitions(SOPHUS_USE_BASIC_LOGGING=1)


include_directories ("${PROJECT_SOURCE_DIR}")

set(BaSpaCho_sources
    baspacho/CoalescedBlockMatrix.cpp
    baspacho/EliminationTree.cpp
    baspacho/MatOpsBlas.cpp
    baspacho/MatOpsRef.cpp
    baspacho/Solver.cpp
    baspacho/SparseStructure.cpp
    baspacho/Utils.cpp)
if(BASPACHO_USE_CUBLAS)
  list(APPEND BaSpaCho_sources
       baspacho/CublasError.cpp
       baspacho/MatOpsCuda.cu)
endif()
add_library(BaSpaCho ${BaSpaCho_sources})
target_link_libraries (BaSpaCho dispenso ${BLAS_LIBRARIES})

if(BASPACHO_USE_CUBLAS)
  set_target_properties(BaSpaCho PROPERTIES CUDA_ARCHITECTURES ${BASPACHO_CUDA_ARCHITECTURES})
  target_link_libraries (BaSpaCho CUDA::cublas CUDA::cusolver)
  target_compile_options(BaSpaCho PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
                        ${BASPACHO_CXX_FLAGS}
                        >)
  target_compile_options(BaSpaCho PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        ${CUDA_NVCC_FLAGS}
                        >)
endif()

if(HAVE_AMD)
  target_link_libraries (BaSpaCho ${AMD_LIBRARIES})
endif()

add_library(testing
            baspacho/testing/TestingMatGen.cpp
            baspacho/testing/TestingUtils.cpp)

target_link_libraries (testing BaSpaCho)


set(BAL_bench_sources
    baspacho/benchmarking/BaAtLarge.cpp
    baspacho/benchmarking/BaAtLargeBench.cpp)
set(bench_sources
    baspacho/benchmarking/Bench.cpp)

if(HAVE_CHOLMOD)
  list(APPEND BAL_bench_sources baspacho/benchmarking/BenchCholmod.cpp)
  list(APPEND bench_sources baspacho/benchmarking/BenchCholmod.cpp)
endif()

add_executable(BAL_bench ${BAL_bench_sources})
target_link_libraries (BAL_bench BaSpaCho testing)

add_executable(bench ${bench_sources})
target_link_libraries (bench BaSpaCho testing)

if(HAVE_CHOLMOD)
  target_link_libraries (BAL_bench ${CHOLMOD_LIBRARIES})
  target_link_libraries (bench ${CHOLMOD_LIBRARIES})
endif()


# unit tests
enable_testing()

function(add_baspacho_test TARGET)
  add_executable(
    "${TARGET}"
    "${ARGN}"
  )
  target_link_libraries(
    "${TARGET}" BaSpaCho testing gtest_main gmock
  )
  gtest_discover_tests("${TARGET}")
endfunction()

add_baspacho_test(
  AccessorTest
  baspacho/tests/AccessorTest.cpp
)
add_baspacho_test(
  CoalescedBlockMatrixTest
  baspacho/tests/CoalescedBlockMatrixTest.cpp
)
add_baspacho_test(
  SparseStructureTest
  baspacho/tests/SparseStructureTest.cpp
)
add_baspacho_test(
  EliminationTreeTest
  baspacho/tests/EliminationTreeTest.cpp
)
add_baspacho_test(
  FactorTest
  baspacho/tests/FactorTest.cpp
)
add_baspacho_test(
  SolveTest
  baspacho/tests/SolveTest.cpp
)
add_baspacho_test(
  MathUtilsTest
  baspacho/tests/MathUtilsTest.cpp
)

if(BASPACHO_USE_CUBLAS)
  add_baspacho_test(
    CudaFactorTest
    baspacho/tests/CudaFactorTest.cpp
  )
  add_baspacho_test(
    CudaSolveTest
    baspacho/tests/CudaSolveTest.cpp
  )
  add_baspacho_test(
    BatchedCudaFactorTest
    baspacho/tests/BatchedCudaFactorTest.cpp
  )
  add_baspacho_test(
    BatchedCudaSolveTest
    baspacho/tests/BatchedCudaSolveTest.cpp
  )
endif()