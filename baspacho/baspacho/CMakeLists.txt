
set(BaSpaCho_sources
    CoalescedBlockMatrix.cpp
    EliminationTree.cpp
    MatOpsBlas.cpp
    MatOpsRef.cpp
    Solver.cpp
    SparseStructure.cpp
    Utils.cpp)
if(BASPACHO_USE_CUBLAS)
    list(APPEND BaSpaCho_sources
         CublasError.cpp
         MatOpsCuda.cu)
endif()

add_library(BaSpaCho
            ${BaSpaCho_sources})

target_link_libraries(BaSpaCho
                      dispenso
                      ${BLAS_LIBRARIES})

if(BASPACHO_USE_CUBLAS)
    target_link_libraries(BaSpaCho
                          CUDA::cublas
                          CUDA::cusolver)
    set_target_properties(BaSpaCho PROPERTIES CUDA_ARCHITECTURES ${BASPACHO_CUDA_ARCHITECTURES})
    target_compile_options(BaSpaCho PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
                           ${BASPACHO_CXX_FLAGS}
                           >)
    target_compile_options(BaSpaCho PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                           ${CUDA_NVCC_FLAGS}
                           >)
endif()
                     
if(HAVE_SUITESPARSE_AMD)
    target_link_libraries(BaSpaCho
                          ${SUITESPARSE_AMD_LIBRARIES})
endif()