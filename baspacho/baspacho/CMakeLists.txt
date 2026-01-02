
set(BaSpaCho_sources
    CoalescedBlockMatrix.cpp
    ComputationModel.cpp
    EliminationTree.cpp
    MatOpsFast.cpp
    MatOpsRef.cpp
    Solver.cpp
    SparseStructure.cpp
    Utils.cpp)

if(BASPACHO_USE_CUBLAS)
    # separate recent and legacy cuda archs (which need atomicAdd's workaround)
    set(BASPACHO_LEGACY_CUDA_ARCHITECTURES "")
    set(BASPACHO_RECENT_CUDA_ARCHITECTURES "")
    foreach(ARCH IN LISTS BASPACHO_CUDA_ARCHITECTURES)
        if(ARCH LESS 60)
            list(APPEND BASPACHO_LEGACY_CUDA_ARCHITECTURES "${ARCH}")
        else()
            list(APPEND BASPACHO_RECENT_CUDA_ARCHITECTURES "${ARCH}")
        endif()
    endforeach()

    list(APPEND BaSpaCho_sources CublasError.cpp)
    if(BASPACHO_RECENT_CUDA_ARCHITECTURES)
        list(APPEND BaSpaCho_sources MatOpsCuda.cu)
    endif()
endif()

if(BASPACHO_USE_METAL)
    # Metal backend uses Objective-C++ (.mm files)
    # We compile these as CXX with explicit -x objective-c++ flag
    # instead of using enable_language(OBJCXX), which would affect
    # the entire project and break embedding in projects with .m files.
    set(BASPACHO_METAL_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/MetalDefs.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/MatOpsMetal.mm)

    # Set .mm files to be treated as CXX with Objective-C++ mode
    set_source_files_properties(${BASPACHO_METAL_SOURCES}
        PROPERTIES
        LANGUAGE CXX
        COMPILE_FLAGS "-x objective-c++ -std=c++17")

    list(APPEND BaSpaCho_sources
         MetalDefs.mm
         MatOpsMetal.mm)
endif()

add_library(${BASPACHO_LIBRARY}  ${BaSpaCho_sources})
set_property(TARGET ${BASPACHO_LIBRARY} PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(${BASPACHO_LIBRARY}
                      dispenso
                      ${BLAS_LIBRARIES})

if(BASPACHO_USE_CUBLAS)
    # if compiling for legacy, create a target for Cuda code with enabled workaround
    if(BASPACHO_LEGACY_CUDA_ARCHITECTURES)
        add_library(BaSpaCho_legacy_cuda_archs OBJECT MatOpsCuda.cu)
        set_target_properties(BaSpaCho_legacy_cuda_archs PROPERTIES CUDA_ARCHITECTURES "${BASPACHO_LEGACY_CUDA_ARCHITECTURES}")
        target_compile_options(BaSpaCho_legacy_cuda_archs PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${BASPACHO_CXX_FLAGS}>)
        target_compile_options(BaSpaCho_legacy_cuda_archs PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
        target_compile_definitions(BaSpaCho_legacy_cuda_archs PRIVATE CUDA_DOUBLE_ATOMIC_ADD_WORKAROUND)
        target_link_libraries(${BASPACHO_LIBRARY} BaSpaCho_legacy_cuda_archs)
    endif()

    target_link_libraries(${BASPACHO_LIBRARY}
                          CUDA::cudart
                          CUDA::cublas
                          CUDA::cusolver)
    if(BASPACHO_RECENT_CUDA_ARCHITECTURES)
        set_target_properties(${BASPACHO_LIBRARY} PROPERTIES CUDA_ARCHITECTURES "${BASPACHO_RECENT_CUDA_ARCHITECTURES}")
    endif()
    target_compile_options(${BASPACHO_LIBRARY} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
endif()

if(BASPACHO_USE_METAL)
    # Compile Metal shaders to metallib
    set(METAL_SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/MetalKernels.metal")
    set(METAL_SHADER_AIR "${CMAKE_CURRENT_BINARY_DIR}/MetalKernels.air")
    set(METAL_SHADER_LIB "${CMAKE_CURRENT_BINARY_DIR}/MetalKernels.metallib")

    add_custom_command(
        OUTPUT "${METAL_SHADER_AIR}"
        COMMAND xcrun -sdk macosx metal -c "${METAL_SHADER_SOURCE}" -o "${METAL_SHADER_AIR}" -std=macos-metal2.4
        DEPENDS "${METAL_SHADER_SOURCE}"
        COMMENT "Compiling Metal shaders to AIR"
    )

    add_custom_command(
        OUTPUT "${METAL_SHADER_LIB}"
        COMMAND xcrun -sdk macosx metallib "${METAL_SHADER_AIR}" -o "${METAL_SHADER_LIB}"
        DEPENDS "${METAL_SHADER_AIR}"
        COMMENT "Linking Metal shader library"
    )

    add_custom_target(metal_shaders ALL DEPENDS "${METAL_SHADER_LIB}")
    add_dependencies(${BASPACHO_LIBRARY} metal_shaders)

    # Link Metal frameworks
    target_link_libraries(${BASPACHO_LIBRARY}
                          ${METAL_FRAMEWORK}
                          ${MPS_FRAMEWORK}
                          ${FOUNDATION_FRAMEWORK})

    # Add ARC flag to the Metal source files
    # Note: -x objective-c++ and -std=c++17 are already set above
    set_source_files_properties(${BASPACHO_METAL_SOURCES}
        PROPERTIES COMPILE_FLAGS "-x objective-c++ -std=c++17 -fobjc-arc")

    # Define path to compiled metallib for runtime loading
    target_compile_definitions(${BASPACHO_LIBRARY} PRIVATE
        BASPACHO_METAL_LIBRARY_PATH="${METAL_SHADER_LIB}")
endif()

target_compile_options(${BASPACHO_LIBRARY} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${BASPACHO_CXX_FLAGS}>)
                     
if(HAVE_SUITESPARSE_AMD)
    target_link_libraries(${BASPACHO_LIBRARY}
                          ${SUITESPARSE_AMD_LIBRARIES})
endif()

if(NOT BUILD_SHARED_LIBS)
    bundle_static_library(${BASPACHO_LIBRARY} BaSpaCho)
endif()
