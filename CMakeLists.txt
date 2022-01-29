cmake_minimum_required(VERSION 3.20)

# set the project name
project(BaSpaCho)
set(CMAKE_CXX_STANDARD 17)

# a few possibilities are:
# * ATLAS
# * OpenBLAS
# * Intel10_64{i}lp{_seq}
#   (if 'i' indices are 64bits, if '_seq' sequential ie single thread)
set(BLA_VENDOR OpenBLAS CACHE STRING
    "BLAS Library to be searched, possibilities are: ATLAS, OpenBLAS, Intel10_64ilp_seq, etc")
message("BLA_VENDOR = " ${BLA_VENDOR})
find_package(BLAS REQUIRED)
message("BLAS linkerFlags = " ${BLAS_LINKER_FLAGS})
message("BLAS Libraries = " ${BLAS_LIBRARIES})
if(BLA_VENDOR MATCHES "Intel10_*")
  message("BLAS appears to be MKL, MKL_ROOT should be set")
  message("Adding includes:" $ENV{MKLROOT}/include)
  include_directories($ENV{MKLROOT}/include) # if MKL
  add_compile_definitions(BASPACHO_USE_MKL)
endif()


# cholmod (for benchmarking comparison)
if(false) # use local installation dir
  find_path(CHOLMOD_INCLUDE_DIR NAMES cholmod.h
            PATHS $ENV{CHOLMODDIR}/Include
            NO_DEFAULT_PATH
            REQUIRED)
  find_library(CHOLMOD_LIBRARIES cholmod
            PATHS $ENV{CHOLMODDIR}/Lib
            NO_DEFAULT_PATH
            REQUIRED)
else()
find_path(CHOLMOD_INCLUDE_DIR NAMES cholmod.h
          PATHS $ENV{CHOLMODDIR} ${INCLUDE_INSTALL_DIR}
          PATH_SUFFIXES suitesparse ufsparse
          REQUIRED)
find_library(CHOLMOD_LIBRARIES cholmod
             PATHS $ENV{CHOLMODDIR} ${LIB_INSTALL_DIR}
             REQUIRED)
endif()
include_directories(${CHOLMOD_INCLUDE_DIR})
message("CHOLMODDIR = " $ENV{CHOLMODDIR})
message("Cholmod Source = " ${CHOLMOD_INCLUDE_DIR})
message("Cholmod Libraries = " ${CHOLMOD_LIBRARIES})

# amd
find_path(AMD_INCLUDE_DIR NAMES amd.h
          PATHS $ENV{AMDDIR} ${INCLUDE_INSTALL_DIR}
          PATH_SUFFIXES suitesparse ufsparse
          REQUIRED)
find_library(AMD_LIBRARIES amd
             PATHS $ENV{AMDDIR} ${LIB_INSTALL_DIR}
             REQUIRED)
include_directories(${AMD_INCLUDE_DIR})
message("Amd Source = " ${AMD_INCLUDE_DIR})
message("Amd Libraries = " ${AMD_LIBRARIES})

# use fetchcontent to download dependencies
include(FetchContent)
cmake_policy(SET CMP0077 NEW)

# dispenso for multithreading
# currently in facebookincubator, soon to be opensourced
FetchContent_Declare(
  dispenso
  GIT_REPOSITORY git@github.com:facebookincubator/dispenso.git
  GIT_TAG 22ff47d07c32303a5059bfec26304a68dff01e65
	)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
FetchContent_MakeAvailable(dispenso)
unset(CMAKE_POSITION_INDEPENDENT_CODE)
include_directories(${dispenso_SOURCE_DIR} ${dispenso_SOURCE_DIR}/dispenso/third-party/moodycamel)
message("Dispenso Source = " ${dispenso_SOURCE_DIR})
message("Dispenso Library Dir = " ${dispenso_BINARY_DIR})

# glog
FetchContent_Declare(
  glog
  URL https://github.com/google/glog/archive/refs/tags/v0.5.0.zip
	)
set(BUILD_TESTING 0 CACHE BOOL "")
FetchContent_MakeAvailable(glog)
message("Glog Source = " ${glog_SOURCE_DIR})
message("Glog Library Dir = " ${glog_BINARY_DIR})


# gtest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
	)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
message("GTest Source = " ${gtest_SOURCE_DIR})
message("GTest Library Dir = " ${gtest_BINARY_DIR})
message("GMock Source = " ${gmock_SOURCE_DIR})
message("GMock Library Dir = " ${gmock_BINARY_DIR})
include(GoogleTest)


# eigen
FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
	)
if(NOT eigen_POPULATED)
  FetchContent_Populate(eigen)
endif()
message("Eigen Source = " ${eigen_SOURCE_DIR})
message("Eigen Library Dir = " ${eigen_BINARY_DIR})
include_directories(${eigen_SOURCE_DIR})


# sophus (for testing on BA problem)
FetchContent_Declare(
  sophus
  GIT_REPOSITORY https://github.com/strasdat/Sophus.git
  GIT_TAG 49a7e1286910019f74fb4f0bb3e213c909f8e1b7
)
if(NOT sophus_POPULATED)
  FetchContent_Populate(sophus)
endif()
message("Sophus Source = " ${sophus_SOURCE_DIR})
message("Sophus Library Dir = " ${sophus_BINARY_DIR})
include_directories(${sophus_SOURCE_DIR})
add_compile_definitions(SOPHUS_USE_BASIC_LOGGING=1)

add_library(BaSpaCho
            baspacho/CoalescedBlockMatrix.cpp
            baspacho/EliminationTree.cpp
            baspacho/MatOpsBlas.cpp
            baspacho/MatOpsRef.cpp
            baspacho/Solver.cpp
            baspacho/SparseStructure.cpp
            baspacho/Utils.cpp)
target_link_libraries (BaSpaCho glog::glog dispenso ${AMD_LIBRARIES} ${CHOLMOD_LIBRARIES})

# IF have blas
target_link_libraries (BaSpaCho ${BLAS_LIBRARIES})

add_library(testing
            testing/TestingMatGen.cpp
            testing/TestingUtils.cpp)

target_link_libraries (testing BaSpaCho)

add_executable(BAL_test
               benchmarking/BaAtLarge.cpp
               benchmarking/BaAtLargeTest.cpp
               benchmarking/BenchCholmod.cpp)
target_link_libraries (BAL_test BaSpaCho testing)

add_executable(bench
               benchmarking/Bench.cpp 
               benchmarking/BenchCholmod.cpp)
target_link_libraries (bench BaSpaCho testing)


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
  CoalescedBlockMatrixTest
  baspacho/tests/CoalescedBlockMatrixTest.cpp
)
add_baspacho_test(
  ExperimentalFactorTest
  experiments/Factor.cpp
  experiments/FactorTest.cpp
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