
add_baspacho_test(AccessorTest AccessorTest.cpp)
add_baspacho_test(CoalescedBlockMatrixTest CoalescedBlockMatrixTest.cpp)
add_baspacho_test(CreateSolverTest CreateSolverTest.cpp)
add_baspacho_test(SparseStructureTest SparseStructureTest.cpp)
add_baspacho_test(EliminationTreeTest EliminationTreeTest.cpp)
add_baspacho_test(FactorTest FactorTest.cpp)
add_baspacho_test(SolveTest SolveTest.cpp)
add_baspacho_test(MathUtilsTest MathUtilsTest.cpp)
add_baspacho_test(PartialFactorSolveTest PartialFactorSolveTest.cpp)

if(BASPACHO_USE_CUBLAS)
add_baspacho_test(CudaFactorTest CudaFactorTest.cpp)
add_baspacho_test(CudaSolveTest CudaSolveTest.cpp)
add_baspacho_test(BatchedCudaFactorTest BatchedCudaFactorTest.cpp)
add_baspacho_test(BatchedCudaSolveTest BatchedCudaSolveTest.cpp)
add_baspacho_test(CudaPartialTest CudaPartialTest.cpp)
endif()