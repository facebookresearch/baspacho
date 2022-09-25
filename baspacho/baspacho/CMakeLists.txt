
set(BaSpaCho_sources
    CoalescedBlockMatrix.cpp
    ComputationModel.cpp
    EliminationTree.cpp
    MatOpsRef.cpp
    Solver.cpp
    SparseStructure.cpp
    Utils.cpp)
if(BASPACHO_USE_BLAS)
    list(APPEND BaSpaCho_sources
         MatOpsBlas.cpp)
endif()
if(BASPACHO_USE_CUBLAS)
    list(APPEND BaSpaCho_sources
         CublasError.cpp
         MatOpsCuda.cu)
endif()

add_library(${BASPACHO_LIBRARY}  ${BaSpaCho_sources})
set_property(TARGET ${BASPACHO_LIBRARY} PROPERTY POSITION_INDEPENDENT_CODE ON)

message("ADDING LIBS: " ${BLAS_LIBRARIES})
target_link_libraries(${BASPACHO_LIBRARY}
                      dispenso
                      ${BLAS_LIBRARIES})

if(BASPACHO_USE_CUBLAS)
    target_link_libraries(${BASPACHO_LIBRARY}
                          CUDA::cudart
                          CUDA::cublas
                          CUDA::cusolver)
    set_target_properties(${BASPACHO_LIBRARY} PROPERTIES CUDA_ARCHITECTURES "${BASPACHO_CUDA_ARCHITECTURES}")
    target_compile_options(${BASPACHO_LIBRARY} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
                           ${BASPACHO_CXX_FLAGS}
                           >)
    target_compile_options(${BASPACHO_LIBRARY} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                           ${CUDA_NVCC_FLAGS}
                           >)
endif()
                     
if(HAVE_SUITESPARSE_AMD)
    target_link_libraries(${BASPACHO_LIBRARY}
                          ${SUITESPARSE_AMD_LIBRARIES})
endif()

if(NOT BUILD_SHARED_LIBS)
    bundle_static_library(${BASPACHO_LIBRARY} BaSpaCho)
endif()