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
cmake --build build -- -j16 && ctest --test-dir build
```
Benchmarking all configurations (using CHOLMOD as baseline):
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

## Benchmarking and fine-tuning
BaSpaCho uses models of the timings of BLAS operations in order to decide on the best node merges
selected by the supernodal algorithm. The trade-offs might change dramatically depending on the
architecture, the number of cores, the BLAS implementation used and CPU vs GPU. Timings required
to fit a model can be collected with the `bench` tool when provided the `-Z` command line argument.
Here is a description of the benchmarking tools provided.

### `bench`
The tool `build/baspacho/benchmarking/bench` can be used to compare the speed of different configurations
on a varied set of synthetic problems. For instance if CHOLMOD is installed you can use it a reference
sparse solver to compare to, and the following compares the most time-consuming operation `factor` using
CHOLMOD as baseline:
```
build/baspacho/benchmarking/bench -B 1_CHOLMOD
```
(other operations are `analysis` and `solve-X` where X can be any number of right-hand sides).
Running  with `-h` gives a view of all operations, solvers and problems available
(and arguments accepting regular expressions in order to test a subselection of solvers/problems).
When running default `factor` operation with `-Z` statistics are saved to files of the form
`<config>_<op>.csv` where the `<config>` looks like `stats_cpu_f8` and discriminates the configuration,
while `<op>` is one of the underlying blas-like operations used in factorization (ie one of `potrf`, `trsm`,
`syge`, `asmbl`, where `syge` stands for `syrk/gemm` for people familiar with BLAS).
The operations generated with a specific configuration (eg `cpu_f64`) can be processed as
```
build/baspacho/examples/opt_comp_model -p stats_cpu_f64_potrf.csv -a stats_cpu_f64_asmbl.csv 
                                       -t stats_cpu_f64_trsm.csv -g stats_cpu_f64_syge.csv
```
on order to fit a computation model that can be provided as a setting to `createSolver` in order
to fine-tune for your architecture.

### `BAL_bench`
It's possible to benchmark on a bundle-adjustment problem from
[Bundle Adjustment in the Large](https://grail.cs.washington.edu/projects/bal/) as
```
build/baspacho/benchmarking/BAL_bench ~/BAL/problem-871-527480-pre.txt
```
ie. the solver is tested on a linear problem which is identical to the 
Again, if installed, CHOLMOD is tested as a baseline, but only on the 'reduced'
camera-camera problem. BaSpaCho is tested on both the point elimination and the
reduced problem.


## Examples
A few examples are provided to illustrate what the library is capable of.

* `Optimizer.h` provides a fully featured Levenberg-Marquardt optimizer, with support for direct solver
and flavors of mixed direct/iterative solvers. The class is used in
  - `OptimizeSimple.cpp`, solving a simple problem of points connected with springs
  - `OptimizeBaAtLarge.cpp`, solving a bundle adjustment problem from [Bundle Adjustment in the Large](https://grail.cs.washington.edu/projects/bal/)
  - `OptimizeCompModel.cpp`, where a computation model params are fit to timings of BLAS operations.
* `PCG_Sample.cpp` contains a demonstration of how it's possible to perform a partial elimination
and complete the solution of a linear system iteratively using PCG.


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
- [X] Fully featured optimizer (direct and iterative solvers), with demo on BAL data

### Longer term todo:
- [X] better heuristics in Node merge, test more configurations
- [X] simple (non-coalesced) symmetric block matrix with mat-vec op, and iterative solver,
      possibly with mixed-precision preconditioner (fast "rough" factor as float, iterate
      on a double vector for improved precision)
- [ ] handle singularity reporting size of non-SPD minor, possibly discard non-definite sparse-elim
      diagonal blocks 
- [ ] optimize analysis and solve steps
- [ ] investigate possible support of update/downdate ops (note that such support in CHOLMOD
      is only for simplicial decomposition, or decompositions are automatically converted to
      simplicial)


## Caveats
* Only supernodal Cholesky is implemented. We focused on problems having at least a certain degree of
interconnectedness, which will benefit of Blas libraries and parallelism. If working with a problem where
simplicial would have better performance (eg banded matrix) just use Cholmod or Eigen. Also, there is no
support for update/downdate operations of factors. Notice that given the strongly sequential nature of
simplicial decomposition the GPU will not be a good fit either.
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