
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
    target_compile_options(${BASPACHO_LIBRARY} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${BASPACHO_CXX_FLAGS}>)
    target_compile_options(${BASPACHO_LIBRARY} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
endif()
                     
if(HAVE_SUITESPARSE_AMD)
    target_link_libraries(${BASPACHO_LIBRARY}
                          ${SUITESPARSE_AMD_LIBRARIES})
endif()

if(NOT BUILD_SHARED_LIBS)
    bundle_static_library(${BASPACHO_LIBRARY} BaSpaCho)
endif()