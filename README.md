# BaSpaCho

BaSpaCho (**Ba**tched **Spa**rse **Cho**lesky) is a state of the art direct solver for symmetric
positive-definite sparse matrices.
It uses supernodal Cholesky decomposition to achieve state of the art performance, turning
portions of the sparse matrix into dense blocks and invoking high-performance BLAS/lapack
libraries. It is designed with optimization libraries in mind, and aims at reducing part of
the complexity in optimization libraries offering the best tool for the job.
Compared to the library currently considered state of the art (CHOLMOD from SuiteSparse) it
supports:
* **pure-CUDA mode with support for batching,** ie. solving a batch of matrices with identical
structure. This is to support differentiable optimization in Theseus library.
* **parallel elimination of independent sparse small elimination nodes.** This is what is essentially
the operation done via "Schur-elimination trick" in mathematical optimization libraries such as Ceres.
This is essentially a workaround to the supernodal algorithm in CHOLMOD being a bad fit for the problem
structure, so that the solver is fed a partially computed decomposition. The Schur-elimination trick
and Cholesky decomposition are essentially the same thing, and it makes sense to have a solver library
detect (or being hinted) the cases where the supernodal algorithm would not perform well, so that a
different strategy can be used.
* **Block-structured matrix data types,** and facilities for working with them. This is another
facility for the benefit of Optimization libraries, which currently have internal data
types for block matrices, but turn the matrices in non-block csr format to feed the data
to CHOLMOD. Skipping this step allows to avoid unneeded complexity and duplication of data.

## Requirements:

* BLAS (ATLAS/OpenBLAS/MKL, see below)

Libraries fetched automatical by build:
* gtest
* Eigen
* Sophus (only used in BA demo)

Optional libraries:
* CUDA toolkit (tested with CUDA 11.5.2/11.6.1), if not available must explicitly disable GPU support, see below.
* AMD, from SuiteSparse, can be used instead of Eigen for reordering algorithm.
* CHOLMOD, from SuiteSparse, used in benchmark as a reference for performance of sparse solvers.

## Configure and Install

Configuring with system blas (eg OpenBLAS):
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```
Configuring with MKL:
```
. /opt/intel/oneapi/setvars.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBLA_VENDOR=Intel10_64lp
```
Compiling and testing:
```
cmake --build build -v -- -j16 && ctest --test-dir build
```
Benchmarking (using CHOLMOD as baseline):
```
build/baspacho/benchmarking/bench -B 1_CHOLMOD
```
Benchmarking on a problem from [Bundle Adjustment in the Large](https://grail.cs.washington.edu/projects/bal/)
```
build/baspacho/benchmarking/BAL_bench ~/BAL/problem-871-527480-pre.txt
```
Show tests:
```
ctest --test-dir build --show-only
```

### Cuda
Cuda is enabled by default with BASPACHO_USE_CUBLAS option (on by default), add
`-DBASPACHO_USE_CUBLAS=0` to disable in build.
May have to add `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` to allow build
to find the cuda compiler.

### Blas

The library used is specified in the CMake variable BLA_VENDOR,
a few possibilities are:
* ATLAS
* OpenBLAS
* Intel10_64{i}lp{_seq}
(if 'i' indices are 64bits, if '_seq' sequential ie single thread)

For the full list check CMake docs at:
https://cmake.org/cmake/help/latest/module/FindBLAS.html#blas-lapack-vendors

### Reordering algorithm Approximate Minimum Degree (AMD)

BaSpaCho can use either the implementation in Eigen of AMD (by default), or the version in library
AMD as part of SuiteSparse. Add `-DBASPACHO_USE_SUITESPARSE_AMD=1` to the build step to use the
implementation in SuiteSparse instead of Eigen.

## Todo
- [ ] add continuous integration for testing
- [ ] document more code (before I forget how it works), and polish API
- [X] float/double in all backends
- [X] Cuda: include and test sparse elimination in BAL_bench
- [X] Cuda: test atomics vs magic-locked block updates (result: magic lock of blocks does not work
      in Cuda unless very expensive synchronized ops are used, making it not competitive)
- [X] Cuda: solve methods
- [X] Cuda: batched mode
- [X] implement specialized solve for the case of "sparse elimination ranges" where the current
      per-supernode solve would be quite slow
- [X] Benchmark results (bench + BAL comparing sparse elimination with optimizers)
- [ ] small optimizer demo on BAL data

### Longer term todo:
- [ ] handle singularity reporting size of non-SPD minor, possibly discard non-definite sparse-elim
      diagonal blocks 
- [ ] better heuristics in Node merge, test more configurations
- [ ] optimize analysis and solve steps
- [ ] simple (non-coalesced) symmetric block matrix with mat-vec op, and iterative solver,
      possibly with mixed-precision preconditioner (fast "rough" factor as float, iterate
      on a double vector for improved precision)
- [ ] investigate possible support of update/downdate ops (note that such support in CHOLMOD
      is only for simplicial decomposition, or decompositions are automatically converted to
      simplicial)

## Caveats

* Only supernodal Cholesky is implemented. We focused on problems having at least a certain degree of
interconnectedness, which will benefit of Blas libraries and parallelism. If working with a problem where
simplicial would have better performance (eg banded matrix) just use Cholmod or Eigen. Also, because of
this there is no support for update/downdate operations of factors. Notice that given the strongly sequential
nature of simplicial decomposition the GPU will not be a good fit either.
* Only reordering method currently supported is Approximate Minimum Degree (AMD), this will probably be
expanded and made more customizable in the future but this is the current status of things.
* The block-structured type of matrices is builtin in the library, and while this presents advantages in
common circumstances the library may prove slighly slower if the matrix lacks completely a block structure,
as the entries will be represented as 1x1 blocks. This is hardly a problem in most practical problems
because matrices naturally have a block-structure depending on parameters of dimensions >1. Also, thread and
cuda-kernel operations are designed around blocks so it's not ideal if you have some huge parameter blocks,
the library will work best when the parameter blocks have sizes 1 to 12 (in a factor graph, generally you
have many parameter blocks of the same type).
* About determinism: assuming BLAS is deterministic, BaSpaCho will be 100% deterministic on the CPU, but
not on CUDA if there is any "sparse elimination" set of parameters, because both factor and solve operations
use atomic addition for parallelism on the GPU. Also, a Cuda architecture >=6 is needed for atomicAdd
on double numbers (this is the compute hardware architecture and not the version of Cuda, arch >=6 means
you need Tesla P100 or GTX 1080-family, or newer. See
[Cuda Architectures](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)).
Otherwise you will have to add define `CUDA_DOUBLE_ATOMIC_ADD_WORKAROUND` in order to enable the workaround
in `CudaAtomic.cuh`.