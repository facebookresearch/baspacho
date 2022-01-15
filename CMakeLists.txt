cmake_minimum_required(VERSION 3.20)

# set the project name
project(BaSpaCho)
set(CMAKE_CXX_STANDARD 17)


# cholmod
# find_path(CHOLMOD_INCLUDES NAMES cholmod.h
#           PATHS $ENV{CHOLMODDIR} ${INCLUDE_INSTALL_DIR}
#           PATH_SUFFIXES suitesparse ufsparse
#           REQUIRED)
# find_library(CHOLMOD_LIBRARIES cholmod
#              PATHS $ENV{CHOLMODDIR} ${LIB_INSTALL_DIR}
#              REQUIRED)
# message("Cholmod Source = " ${CHOLMOD_INCLUDES})
# message("Cholmod Library Dir = " ${CHOLMOD_LIBRARIES})

# amd
find_path(AMD_INCLUDES NAMES amd.h
          PATHS $ENV{CHOLMODDIR} ${INCLUDE_INSTALL_DIR}
          PATH_SUFFIXES suitesparse ufsparse
          REQUIRED)
find_library(AMD_LIBRARIES amd
             PATHS $ENV{CHOLMODDIR} ${LIB_INSTALL_DIR}
             REQUIRED)
message("Amd Source = " ${AMD_INCLUDES})
message("Amd Library Dir = " ${AMD_LIBRARIES})

include(FetchContent)

# glog
cmake_policy(SET CMP0077 NEW)
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
            BlockMatrix.cpp
            EliminationTree.cpp
            Factor.cpp
            SparseStructure.cpp
            TestingUtils.cpp
            Utils.cpp)
target_link_libraries (BaSpaCho glog::glog amd)


add_executable(BAL_test BaAtLarge.cpp BaAtLargeTest.cpp)
target_link_libraries (BAL_test glog::glog BaSpaCho)


enable_testing()

add_executable(
  BlockMatrixTest
  BlockMatrixTest.cpp
	)
add_executable(
  FactorTest
  FactorTest.cpp
	)
add_executable(
  SparseStructureTest
  SparseStructureTest.cpp
  )
add_executable(
  EliminationTreeTest
  EliminationTreeTest.cpp
  )
target_link_libraries(
  BlockMatrixTest
	BaSpaCho
  gtest_main
	gmock
)
target_link_libraries(
  FactorTest
	BaSpaCho
  gtest_main
	gmock
)
target_link_libraries(
  SparseStructureTest
	BaSpaCho
  gtest_main
	gmock
)
target_link_libraries(
  EliminationTreeTest
	BaSpaCho
  gtest_main
	gmock
)
gtest_discover_tests(BlockMatrixTest)
gtest_discover_tests(FactorTest)
gtest_discover_tests(SparseStructureTest)
gtest_discover_tests(EliminationTreeTest)
