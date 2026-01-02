#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# FindCLBlast.cmake
# Find CLBlast library for high-performance OpenCL BLAS operations
#
# This module sets the following variables:
#   CLBlast_FOUND        - TRUE if CLBlast was found
#   CLBlast_INCLUDE_DIRS - Include directories for CLBlast
#   CLBlast_LIBRARIES    - CLBlast library
#   CLBLAST_LIBRARIES    - Alias for CLBlast_LIBRARIES (compatibility)
#
# Usage:
#   find_package(CLBlast REQUIRED)
#   target_link_libraries(mytarget ${CLBlast_LIBRARIES})
#   target_include_directories(mytarget ${CLBlast_INCLUDE_DIRS})

find_path(CLBlast_INCLUDE_DIR
    NAMES clblast.h
    HINTS
        $ENV{CLBLAST_ROOT}
        ${CLBLAST_ROOT}
    PATH_SUFFIXES include
    PATHS
        /usr/local
        /usr
        /opt/local
        /opt
)

find_library(CLBlast_LIBRARY
    NAMES clblast
    HINTS
        $ENV{CLBLAST_ROOT}
        ${CLBLAST_ROOT}
    PATH_SUFFIXES lib lib64
    PATHS
        /usr/local
        /usr
        /opt/local
        /opt
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CLBlast
    REQUIRED_VARS CLBlast_LIBRARY CLBlast_INCLUDE_DIR
)

if(CLBlast_FOUND)
    set(CLBlast_INCLUDE_DIRS ${CLBlast_INCLUDE_DIR})
    set(CLBlast_LIBRARIES ${CLBlast_LIBRARY})
    set(CLBLAST_LIBRARIES ${CLBlast_LIBRARY})  # Alias for compatibility

    if(NOT TARGET CLBlast::CLBlast)
        add_library(CLBlast::CLBlast UNKNOWN IMPORTED)
        set_target_properties(CLBlast::CLBlast PROPERTIES
            IMPORTED_LOCATION "${CLBlast_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${CLBlast_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(CLBlast_INCLUDE_DIR CLBlast_LIBRARY)
