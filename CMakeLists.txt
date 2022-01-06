cmake_minimum_required(VERSION 3.20)

# set the project name
project(BaSpaCho)
set(CMAKE_CXX_STANDARD 17)


include(FetchContent)

# glog
FetchContent_Declare(
  glog
  URL https://github.com/google/glog/archive/refs/tags/v0.5.0.zip
	)
if(NOT glog_POPULATED)
  FetchContent_Populate(glog)
  set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE BOOL "")
  add_subdirectory(${glog_SOURCE_DIR} ${glog_BINARY_DIR} EXCLUDE_FROM_ALL)
  unset(CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
endif()
set(glog_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glog)
message("Glog Source = " ${glog_SOURCE_DIR})
message("Glog Library Dir = " ${glog_BINARY_DIR})


# gtest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
	)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE BOOL "")
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
  unset(CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
endif()
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
  set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE BOOL "")
  add_subdirectory(${eigen_SOURCE_DIR} ${eigen_BINARY_DIR} EXCLUDE_FROM_ALL)
  unset(CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
endif()
FetchContent_MakeAvailable(eigen)
message("Eigen Source = " ${eigen_SOURCE_DIR})
message("Eigen Library Dir = " ${eigen_BINARY_DIR})
include_directories(${eigen_SOURCE_DIR})


add_library(BaSpaCho BlockMatrix.cpp)
target_link_libraries (BaSpaCho glog::glog)


enable_testing()

add_executable(
  BlockMatrixTest
  BlockMatrixTest.cpp
	)
# target_include_directories(BlockMatrixTest INTERFACE ${gtest_SOURCE_DIR}/include)
target_link_libraries(
  BlockMatrixTest
	BaSpaCho
  gtest_main
	gmock
)
gtest_discover_tests(BlockMatrixTest)
