/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <chrono>
#include <iostream>
#include <typeindex>

#include <Eigen/Dense>
#include "baspacho/baspacho/CoalescedBlockMatrix.h"
#include "baspacho/baspacho/DebugMacros.h"
#include "baspacho/baspacho/MatOps.h"
#include "baspacho/baspacho/MetalDefs.h"
#include "baspacho/baspacho/Utils.h"

namespace BaSpaCho {

using namespace std;
using hrc = chrono::high_resolution_clock;
using tdelta = chrono::duration<double>;

// Synchronization ops for Metal
struct MetalSyncOps {
  static void sync() { MetalContext::instance().synchronize(); }
};

// Symbolic elimination context for Metal
struct MetalSymElimCtx : SymElimCtx {
  MetalSymElimCtx() {}
  virtual ~MetalSymElimCtx() override {}

  int64_t numColumns;
  int64_t numBlockPairs;
  MetalMirror<int64_t> makeBlockPairEnumStraight;
};

// Forward declarations
struct MetalSymbolicCtx;

template <typename T>
struct MetalNumericCtx;

template <typename T>
struct MetalSolveCtx;

// Symbolic context for Metal operations
struct MetalSymbolicCtx : SymbolicCtx {
  MetalSymbolicCtx(const CoalescedBlockMatrixSkel& skel_, const std::vector<int64_t>& permutation)
      : skel(skel_) {
    @autoreleasepool {
      device = (__bridge id<MTLDevice>)MetalContext::instance().device();
      commandQueue = (__bridge id<MTLCommandQueue>)MetalContext::instance().commandQueue();

      // Load all skeleton data to GPU buffers
      devLumpToSpan.load(skel.lumpToSpan);
      devChainRowsTillEnd.load(skel.chainRowsTillEnd);
      devChainRowSpan.load(skel.chainRowSpan);
      devSpanOffsetInLump.load(skel.spanOffsetInLump);
      devLumpStart.load(skel.lumpStart);
      devChainColPtr.load(skel.chainColPtr);
      devChainData.load(skel.chainData);
      devBoardColPtr.load(skel.boardColPtr);
      devBoardChainColOrd.load(skel.boardChainColOrd);
      devSpanStart.load(skel.spanStart);
      devSpanToLump.load(skel.spanToLump);
      devPermutation.load(permutation);

      NSLog(@"MetalSymbolicCtx initialized");
    }
  }

  virtual ~MetalSymbolicCtx() override {}

  virtual PermutedCoalescedAccessor deviceAccessor() override {
    PermutedCoalescedAccessor retv;
    retv.init(devSpanStart.ptr(), devSpanToLump.ptr(), devLumpStart.ptr(), devSpanOffsetInLump.ptr(),
              devChainColPtr.ptr(), devChainRowSpan.ptr(), devChainData.ptr(), devPermutation.ptr());
    return retv;
  }

  virtual SymElimCtxPtr prepareElimination(int64_t lumpsBegin, int64_t lumpsEnd) override {
    MetalSymElimCtx* elim = new MetalSymElimCtx;

    vector<int64_t> makeStraight(lumpsEnd - lumpsBegin + 1);

    // For each lump, compute number of pairs contributing to elimination
    for (int64_t l = lumpsBegin; l < lumpsEnd; l++) {
      int64_t startPtr = skel.chainColPtr[l] + 1;  // skip diag block
      int64_t endPtr = skel.chainColPtr[l + 1];
      int64_t n = endPtr - startPtr;
      makeStraight[l - lumpsBegin] = n * (n + 1) / 2;
    }
    cumSumVec(makeStraight);

    elim->numColumns = lumpsEnd - lumpsBegin;
    elim->numBlockPairs = makeStraight[makeStraight.size() - 1];
    elim->makeBlockPairEnumStraight.load(makeStraight);

    return SymElimCtxPtr(elim);
  }

  virtual NumericCtxBase* createNumericCtxForType(type_index tIdx, int64_t tempBufSize,
                                                  int batchSize) override;

  virtual SolveCtxBase* createSolveCtxForType(type_index tIdx, int nRHS, int batchSize) override;

  const CoalescedBlockMatrixSkel& skel;

  id<MTLDevice> device;
  id<MTLCommandQueue> commandQueue;

  // Device buffers (mirrors of skeleton data)
  MetalMirror<int64_t> devLumpToSpan;
  MetalMirror<int64_t> devChainRowsTillEnd;
  MetalMirror<int64_t> devChainRowSpan;
  MetalMirror<int64_t> devSpanOffsetInLump;
  MetalMirror<int64_t> devLumpStart;
  MetalMirror<int64_t> devChainColPtr;
  MetalMirror<int64_t> devChainData;
  MetalMirror<int64_t> devBoardColPtr;
  MetalMirror<int64_t> devBoardChainColOrd;
  MetalMirror<int64_t> devSpanStart;
  MetalMirror<int64_t> devSpanToLump;
  MetalMirror<int64_t> devPermutation;
};

// Metal operations factory
struct MetalOps : Ops {
  virtual SymbolicCtxPtr createSymbolicCtx(const CoalescedBlockMatrixSkel& skel,
                                           const std::vector<int64_t>& permutation) override {
    return SymbolicCtxPtr(new MetalSymbolicCtx(skel, permutation));
  }
};

// Helper to dispatch a Metal compute kernel
static void dispatchKernel(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                           void (^encodeBlock)(id<MTLComputeCommandEncoder>), NSUInteger numThreads) {
  @autoreleasepool {
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];
    encodeBlock(encoder);

    // Calculate thread group size
    NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
    threadGroupSize = MIN(threadGroupSize, numThreads);

    MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
    MTLSize numGroups = MTLSizeMake((numThreads + threadGroupSize - 1) / threadGroupSize, 1, 1);

    [encoder dispatchThreadgroups:numGroups threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
  }
}

// Numeric context for float - Metal implementation
template <>
struct MetalNumericCtx<float> : NumericCtx<float> {
  MetalNumericCtx(MetalSymbolicCtx& sym_, int64_t tempBufSize, int64_t numSpans)
      : sym(sym_), numSpans_(numSpans), spanToChainOffset(numSpans) {
    tempBuffer.resizeToAtLeast(tempBufSize);
    devSpanToChainOffset.resizeToAtLeast(numSpans);
  }

  virtual ~MetalNumericCtx() override {}

  virtual void pseudoFactorSpans(float* data, int64_t spanBegin, int64_t spanEnd) override {
    @autoreleasepool {
      // Find the MTLBuffer for data
      auto bufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      if (!bufferInfo.first) {
        throw std::runtime_error("MetalNumericCtx<float>::pseudoFactorSpans: data buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
      size_t dataBaseOffset = bufferInfo.second;

      // Get pipeline state for factor_spans_kernel_float
      id<MTLComputePipelineState> pipeline =
          (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
              "factor_spans_kernel_float");

      int64_t numSpans = spanEnd - spanBegin;
      if (numSpans <= 0) return;

      dispatchKernel(
          sym.commandQueue, pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanToLump.buffer() offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanOffsetInLump.buffer()
                        offset:0
                       atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpToSpan.buffer() offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer() offset:0 atIndex:3];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer() offset:0 atIndex:4];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                        offset:0
                       atIndex:5];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer() offset:0 atIndex:6];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devBoardColPtr.buffer()
                        offset:0
                       atIndex:7];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devBoardChainColOrd.buffer()
                        offset:0
                       atIndex:8];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                        offset:0
                       atIndex:9];
            [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:10];
            [encoder setBytes:&spanBegin length:sizeof(int64_t) atIndex:11];
            [encoder setBytes:&spanEnd length:sizeof(int64_t) atIndex:12];
          },
          (NSUInteger)numSpans);
    }
  }

  virtual void doElimination(const SymElimCtx& elimData, float* data, int64_t lumpsBegin,
                             int64_t lumpsEnd) override {
    @autoreleasepool {
      const MetalSymElimCtx* pElim = dynamic_cast<const MetalSymElimCtx*>(&elimData);
      BASPACHO_CHECK_NOTNULL(pElim);
      const MetalSymElimCtx& elim = *pElim;

      // Find the MTLBuffer for data
      auto bufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      if (!bufferInfo.first) {
        throw std::runtime_error("MetalNumericCtx<float>::doElimination: data buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
      size_t dataBaseOffset = bufferInfo.second;

      int64_t numLumps = lumpsEnd - lumpsBegin;
      if (numLumps <= 0) return;

      // Step 1: Factor lumps (Cholesky on diagonal blocks + below-diagonal solve)
      {
        id<MTLComputePipelineState> pipeline =
            (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
                "factor_lumps_kernel_float");

        dispatchKernel(
            sym.commandQueue, pipeline,
            ^(id<MTLComputeCommandEncoder> encoder) {
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer()
                          offset:0
                         atIndex:0];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                          offset:0
                         atIndex:1];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer()
                          offset:0
                         atIndex:2];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devBoardColPtr.buffer()
                          offset:0
                         atIndex:3];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devBoardChainColOrd.buffer()
                          offset:0
                         atIndex:4];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                          offset:0
                         atIndex:5];
              [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:6];
              [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:7];
              [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:8];
            },
            (NSUInteger)numLumps);
      }

      // Step 2: Sparse elimination
      if (elim.numBlockPairs > 0) {
        id<MTLComputePipelineState> pipeline =
            (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
                "sparse_elim_straight_kernel_float");

        dispatchKernel(
            sym.commandQueue, pipeline,
            ^(id<MTLComputeCommandEncoder> encoder) {
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                          offset:0
                         atIndex:0];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer()
                          offset:0
                         atIndex:1];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                          offset:0
                         atIndex:2];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer()
                          offset:0
                         atIndex:3];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer()
                          offset:0
                         atIndex:4];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanToLump.buffer()
                          offset:0
                         atIndex:5];
              [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanOffsetInLump.buffer()
                          offset:0
                         atIndex:6];
              [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:7];
              [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:8];
              [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:9];
              [encoder setBuffer:(__bridge id<MTLBuffer>)elim.makeBlockPairEnumStraight.buffer()
                          offset:0
                         atIndex:10];
              [encoder setBytes:&elim.numBlockPairs length:sizeof(int64_t) atIndex:11];
            },
            (NSUInteger)elim.numBlockPairs);
      }
    }
  }

  virtual void potrf(int64_t n, float* data, int64_t offA) override {
    @autoreleasepool {
      if (n <= 0) return;

      // Use row-major (matches CpuBaseNumericCtx)
      using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

      Eigen::Map<MatRMaj> matA(data + offA, n, n);
      Eigen::LLT<Eigen::Ref<MatRMaj>> llt(matA);

      if (llt.info() != Eigen::Success) {
        fprintf(stderr, "Metal potrf: Cholesky failed\n");
      }
    }
  }

  virtual void trsm(int64_t n, int64_t k, float* data, int64_t offA, int64_t offB) override {
    @autoreleasepool {
      if (n <= 0 || k <= 0) return;

      // Use row-major for B, column-major for A (matches CpuBaseNumericCtx)
      using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      using MatCMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

      // col-major's upper = (row-major's lower).transpose()
      Eigen::Map<const MatCMaj> matA(data + offA, n, n);
      Eigen::Map<MatRMaj> matB(data + offB, k, n);
      matA.template triangularView<Eigen::Upper>().template solveInPlace<Eigen::OnTheRight>(matB);
    }
  }

  virtual void saveSyrkGemm(int64_t m, int64_t n, int64_t k, const float* data,
                            int64_t offset) override {
    @autoreleasepool {
      if (m <= 0 || n <= 0 || k <= 0) return;

      // Ensure temp buffer is large enough
      tempBuffer.resizeToAtLeast(m * n);

      // Use MPS for larger matrices (threshold based on empirical testing)
      // MPS dispatch overhead makes it slower for small matrices
      static constexpr int64_t kMpsThreshold = 64 * 64 * 64;  // ~262k ops
      bool useMps = (m * n * k >= kMpsThreshold);

      if (useMps) {
        // Use MPS matrix multiplication: C = B * A^T
        // Input: data[offset] is row-major with dims (m, k) for A and (n, k) for B (same memory)
        // Output: tempBuffer is row-major with dims (n, m)
        //
        // MPS uses row-major storage. MPSMatrixMultiplication computes:
        //   result = alpha * op(left) * op(right) + beta * result
        //
        // We want: C(n,m) = B(n,k) * A^T(k,m)
        // So: left=B, right=A with transposeRight=YES

        auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
        if (!dataBufferInfo.first) {
          throw std::runtime_error("MetalNumericCtx<float>::saveSyrkGemm: data buffer not found");
        }
        id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
        size_t dataBaseOffset = dataBufferInfo.second;
        id<MTLBuffer> tempBuf = (__bridge id<MTLBuffer>)tempBuffer.buffer();

        // Matrix A: row-major (m rows, k cols), stride = k
        MPSMatrixDescriptor* descA =
            [MPSMatrixDescriptor matrixDescriptorWithRows:m
                                                  columns:k
                                                 rowBytes:k * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];

        // Matrix B: row-major (n rows, k cols), stride = k
        MPSMatrixDescriptor* descB =
            [MPSMatrixDescriptor matrixDescriptorWithRows:n
                                                  columns:k
                                                 rowBytes:k * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];

        // Matrix C: row-major (n rows, m cols), stride = m
        MPSMatrixDescriptor* descC =
            [MPSMatrixDescriptor matrixDescriptorWithRows:n
                                                  columns:m
                                                 rowBytes:m * sizeof(float)
                                                 dataType:MPSDataTypeFloat32];

        MPSMatrix* mpsA = [[MPSMatrix alloc] initWithBuffer:dataBuffer
                                                     offset:dataBaseOffset + offset * sizeof(float)
                                                 descriptor:descA];
        MPSMatrix* mpsB = [[MPSMatrix alloc] initWithBuffer:dataBuffer
                                                     offset:dataBaseOffset + offset * sizeof(float)
                                                 descriptor:descB];
        MPSMatrix* mpsC = [[MPSMatrix alloc] initWithBuffer:tempBuf offset:0 descriptor:descC];

        // Compute: C(n,m) = B(n,k) * A^T(k,m)
        // left=B, right=A, transposeRight=YES
        MPSMatrixMultiplication* gemm =
            [[MPSMatrixMultiplication alloc] initWithDevice:sym.device
                                              transposeLeft:NO    // B as-is
                                             transposeRight:YES   // A^T
                                                 resultRows:n
                                              resultColumns:m
                                            interiorColumns:k
                                                      alpha:1.0
                                                       beta:0.0];

        id<MTLCommandBuffer> cmdBuf = [sym.commandQueue commandBuffer];
        [gemm encodeToCommandBuffer:cmdBuf leftMatrix:mpsB rightMatrix:mpsA resultMatrix:mpsC];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
      } else {
        // Use Eigen for small matrices (lower overhead)
        using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

        const float* AB = data + offset;
        Eigen::Map<const MatRMaj> matA(AB, m, k);
        Eigen::Map<const MatRMaj> matB(AB, n, k);
        Eigen::Map<MatRMaj> matC(tempBuffer.ptr(), n, m);
        matC.noalias() = matB * matA.transpose();
      }
    }
  }

  virtual void prepareAssemble(int64_t targetLump) override {
    // Prepare chain offset mapping for assembly (same as CUDA version)
    const CoalescedBlockMatrixSkel& skel = sym.skel;

    for (int64_t i = skel.chainColPtr[targetLump], iEnd = skel.chainColPtr[targetLump + 1]; i < iEnd;
         i++) {
      spanToChainOffset[skel.chainRowSpan[i]] = skel.chainData[i];
    }

    // Copy to device
    memcpy(devSpanToChainOffset.ptr(), spanToChainOffset.data(),
           spanToChainOffset.size() * sizeof(int64_t));
  }

  virtual void assemble(float* data, int64_t rectRowBegin, int64_t dstStride,
                        int64_t srcColDataOffset, int64_t srcRectWidth, int64_t numBlockRows,
                        int64_t numBlockCols) override {
    @autoreleasepool {
      if (numBlockRows <= 0 || numBlockCols <= 0) return;

      // Find the MTLBuffer for data
      auto bufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      if (!bufferInfo.first) {
        throw std::runtime_error("MetalNumericCtx<float>::assemble: data buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)bufferInfo.first;
      size_t dataBaseOffset = bufferInfo.second;

      // Get pipeline state
      id<MTLComputePipelineState> pipeline =
          (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
              "assemble_kernel_float");

      int64_t numThreads = numBlockRows * numBlockCols;

      // Compute startRow = chainRowsTillEnd[srcColDataOffset - 1] (element before chain start)
      // This matches CPU ref where startRow = pChainRowsTillEnd[-1] after offsetting the pointer
      int64_t startRow = (srcColDataOffset > 0) ? sym.skel.chainRowsTillEnd[srcColDataOffset - 1] : 0;

      dispatchKernel(
          sym.commandQueue, pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBytes:&numBlockRows length:sizeof(int64_t) atIndex:0];
            [encoder setBytes:&numBlockCols length:sizeof(int64_t) atIndex:1];
            [encoder setBytes:&startRow length:sizeof(int64_t) atIndex:2];
            [encoder setBytes:&srcRectWidth length:sizeof(int64_t) atIndex:3];
            [encoder setBytes:&dstStride length:sizeof(int64_t) atIndex:4];
            // pChainRowsTillEnd offset by srcColDataOffset
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                        offset:srcColDataOffset * sizeof(int64_t)
                       atIndex:5];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                        offset:srcColDataOffset * sizeof(int64_t)
                       atIndex:6];
            [encoder setBuffer:(__bridge id<MTLBuffer>)devSpanToChainOffset.buffer()
                        offset:0
                       atIndex:7];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanOffsetInLump.buffer()
                        offset:0
                       atIndex:8];
            [encoder setBuffer:(__bridge id<MTLBuffer>)tempBuffer.buffer() offset:0 atIndex:9];
            [encoder setBuffer:dataBuffer offset:dataBaseOffset atIndex:10];
          },
          (NSUInteger)numThreads);
    }
  }

  MetalSymbolicCtx& sym;
  int64_t numSpans_;
  MetalMirror<float> tempBuffer;
  MetalMirror<int64_t> devSpanToChainOffset;
  std::vector<int64_t> spanToChainOffset;
};

// Solve context for float - Metal implementation
template <>
struct MetalSolveCtx<float> : SolveCtx<float> {
  MetalSolveCtx(MetalSymbolicCtx& sym_, int nRHS_) : sym(sym_), nRHS(nRHS_) {
    // Allocate temp buffer for vector assembly
    tempVecBuffer.resizeToAtLeast(sym.skel.order() * nRHS);
  }

  virtual ~MetalSolveCtx() override {}

  virtual void sparseElimSolveL(const SymElimCtx& elimData, const float* data, int64_t lumpsBegin,
                                int64_t lumpsEnd, float* C, int64_t ldc) override {
    @autoreleasepool {
      const MetalSymElimCtx* pElim = dynamic_cast<const MetalSymElimCtx*>(&elimData);
      BASPACHO_CHECK_NOTNULL(pElim);

      // Find buffers
      auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      auto cBufferInfo = MetalBufferRegistry::instance().findBuffer(C);
      if (!dataBufferInfo.first || !cBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::sparseElimSolveL: buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
      id<MTLBuffer> cBuffer = (__bridge id<MTLBuffer>)cBufferInfo.first;
      size_t dataOffset = dataBufferInfo.second;
      size_t cOffset = cBufferInfo.second;

      int64_t numLumps = lumpsEnd - lumpsBegin;
      if (numLumps <= 0) return;

      // Dispatch diagonal solve kernel
      id<MTLComputePipelineState> pipeline =
          (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
              "sparseElim_diagSolveL_float");

      int64_t nRHS64 = nRHS;
      dispatchKernel(
          sym.commandQueue, pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer() offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                        offset:0
                       atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer() offset:0 atIndex:2];
            [encoder setBuffer:dataBuffer offset:dataOffset atIndex:3];
            [encoder setBuffer:cBuffer offset:cOffset atIndex:4];
            [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:5];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:6];
            [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:7];
            [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:8];
          },
          (NSUInteger)numLumps);
    }
  }

  virtual void sparseElimSolveLt(const SymElimCtx& elimData, const float* data, int64_t lumpsBegin,
                                 int64_t lumpsEnd, float* C, int64_t ldc) override {
    @autoreleasepool {
      const MetalSymElimCtx* pElim = dynamic_cast<const MetalSymElimCtx*>(&elimData);
      BASPACHO_CHECK_NOTNULL(pElim);

      // Find buffers
      auto dataBufferInfo = MetalBufferRegistry::instance().findBuffer(data);
      auto cBufferInfo = MetalBufferRegistry::instance().findBuffer(C);
      if (!dataBufferInfo.first || !cBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::sparseElimSolveLt: buffer not found");
      }
      id<MTLBuffer> dataBuffer = (__bridge id<MTLBuffer>)dataBufferInfo.first;
      id<MTLBuffer> cBuffer = (__bridge id<MTLBuffer>)cBufferInfo.first;
      size_t dataOffset = dataBufferInfo.second;
      size_t cOffset = cBufferInfo.second;

      int64_t numLumps = lumpsEnd - lumpsBegin;
      if (numLumps <= 0) return;

      // Dispatch diagonal solve kernel
      id<MTLComputePipelineState> pipeline =
          (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
              "sparseElim_diagSolveLt_float");

      int64_t nRHS64 = nRHS;
      dispatchKernel(
          sym.commandQueue, pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devLumpStart.buffer() offset:0 atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainColPtr.buffer()
                        offset:0
                       atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainData.buffer() offset:0 atIndex:2];
            [encoder setBuffer:dataBuffer offset:dataOffset atIndex:3];
            [encoder setBuffer:cBuffer offset:cOffset atIndex:4];
            [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:5];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:6];
            [encoder setBytes:&lumpsBegin length:sizeof(int64_t) atIndex:7];
            [encoder setBytes:&lumpsEnd length:sizeof(int64_t) atIndex:8];
          },
          (NSUInteger)numLumps);
    }
  }

  virtual void symm(const float* data, int64_t offset, int64_t n, const float* C, int64_t offC,
                    int64_t ldc, float* D, int64_t ldd, float alpha) override {
    @autoreleasepool {
      if (n <= 0 || nRHS <= 0) return;

      // Use row-major for data buffer (matches CpuBaseSolveCtx)
      using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      using OuterStride = Eigen::OuterStride<Eigen::Dynamic>;
      using OuterStridedCMajMatK =
          Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     OuterStride>;
      using OuterStridedCMajMatM =
          Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     OuterStride>;

      Eigen::Map<const MatRMaj> matA(data + offset, n, n);
      OuterStridedCMajMatK matC(C + offC, n, nRHS, OuterStride(ldc));
      OuterStridedCMajMatM matD(D, n, nRHS, OuterStride(ldd));
      matD.noalias() += alpha * (MatRMaj(matA.template selfadjointView<Eigen::Lower>()) * matC);
    }
  }

  virtual void solveL(const float* data, int64_t offset, int64_t n, float* C, int64_t offC,
                      int64_t ldc) override {
    @autoreleasepool {
      if (n <= 0 || nRHS <= 0) return;

      // Use row-major for data buffer (matches CpuBaseSolveCtx)
      using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      using OuterStride = Eigen::OuterStride<Eigen::Dynamic>;
      using OuterStridedCMajMatM =
          Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     OuterStride>;

      Eigen::Map<const MatRMaj> matA(data + offset, n, n);
      OuterStridedCMajMatM matC(C + offC, n, nRHS, OuterStride(ldc));
      matA.template triangularView<Eigen::Lower>().solveInPlace(matC);
    }
  }

  virtual void gemv(const float* data, int64_t offset, int64_t nRows, int64_t nCols, const float* A,
                    int64_t offA, int64_t lda, float alpha) override {
    @autoreleasepool {
      if (nRows <= 0 || nCols <= 0 || nRHS <= 0) return;

      // Use row-major for data buffer (matches CpuBaseSolveCtx)
      using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      using OuterStride = Eigen::OuterStride<Eigen::Dynamic>;
      using OuterStridedCMajMatK =
          Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     OuterStride>;

      tempVecBuffer.resizeToAtLeast(nRows * nRHS);

      Eigen::Map<const MatRMaj> matM(data + offset, nRows, nCols);
      OuterStridedCMajMatK matA(A + offA, nCols, nRHS, OuterStride(lda));
      Eigen::Map<MatRMaj> matC(tempVecBuffer.ptr(), nRows, nRHS);
      matC.noalias() = alpha * (matM * matA);
    }
  }

  virtual void assembleVec(int64_t chainColPtr, int64_t numColItems, float* C, int64_t ldc) override {
    @autoreleasepool {
      if (numColItems <= 0) return;

      auto cBufferInfo = MetalBufferRegistry::instance().findBuffer(C);
      if (!cBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::assembleVec: buffer not found");
      }
      id<MTLBuffer> cBuffer = (__bridge id<MTLBuffer>)cBufferInfo.first;
      size_t cOffset = cBufferInfo.second;

      id<MTLComputePipelineState> pipeline =
          (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
              "assembleVec_kernel_float");

      // startRow = chainRowsTillEnd[chainColPtr - 1] (element before chain start)
      int64_t startRow =
          (chainColPtr > 0) ? sym.devChainRowsTillEnd.ptr()[chainColPtr - 1] : 0;

      int64_t nRHS64 = nRHS;
      dispatchKernel(
          sym.commandQueue, pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                        offset:chainColPtr * sizeof(int64_t)
                       atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                        offset:chainColPtr * sizeof(int64_t)
                       atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer() offset:0 atIndex:2];
            [encoder setBuffer:(__bridge id<MTLBuffer>)tempVecBuffer.buffer() offset:0 atIndex:3];
            [encoder setBytes:&numColItems length:sizeof(int64_t) atIndex:4];
            [encoder setBuffer:cBuffer offset:cOffset atIndex:5];
            [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:6];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:7];
            [encoder setBytes:&startRow length:sizeof(int64_t) atIndex:8];
          },
          (NSUInteger)numColItems);
    }
  }

  virtual void solveLt(const float* data, int64_t offset, int64_t n, float* C, int64_t offC,
                       int64_t ldc) override {
    @autoreleasepool {
      if (n <= 0 || nRHS <= 0) return;

      // Use row-major for data buffer (matches CpuBaseSolveCtx)
      using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      using OuterStride = Eigen::OuterStride<Eigen::Dynamic>;
      using OuterStridedCMajMatM =
          Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     OuterStride>;

      Eigen::Map<const MatRMaj> matA(data + offset, n, n);
      OuterStridedCMajMatM matC(C + offC, n, nRHS, OuterStride(ldc));
      matA.template triangularView<Eigen::Lower>().adjoint().solveInPlace(matC);
    }
  }

  virtual void gemvT(const float* data, int64_t offset, int64_t nRows, int64_t nCols, float* A,
                     int64_t offA, int64_t lda, float alpha) override {
    @autoreleasepool {
      if (nRows <= 0 || nCols <= 0 || nRHS <= 0) return;

      // Use row-major for data buffer (matches CpuBaseSolveCtx)
      using MatRMaj = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      using OuterStride = Eigen::OuterStride<Eigen::Dynamic>;
      using OuterStridedCMajMatM =
          Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>, 0,
                     OuterStride>;

      Eigen::Map<const MatRMaj> matM(data + offset, nRows, nCols);
      OuterStridedCMajMatM matA(A + offA, nCols, nRHS, OuterStride(lda));
      Eigen::Map<const MatRMaj> matC(tempVecBuffer.ptr(), nRows, nRHS);
      matA.noalias() += alpha * (matM.transpose() * matC);
    }
  }

  virtual void assembleVecT(const float* C, int64_t ldc, int64_t chainColPtr,
                            int64_t numColItems) override {
    @autoreleasepool {
      if (numColItems <= 0) return;

      auto cBufferInfo = MetalBufferRegistry::instance().findBuffer(C);
      if (!cBufferInfo.first) {
        throw std::runtime_error("MetalSolveCtx<float>::assembleVecT: buffer not found");
      }
      id<MTLBuffer> cBuffer = (__bridge id<MTLBuffer>)cBufferInfo.first;
      size_t cOffset = cBufferInfo.second;

      id<MTLComputePipelineState> pipeline =
          (__bridge id<MTLComputePipelineState>)MetalContext::instance().getPipelineState(
              "assembleVecT_kernel_float");

      // startRow = chainRowsTillEnd[chainColPtr - 1] (element before chain start)
      int64_t startRow =
          (chainColPtr > 0) ? sym.devChainRowsTillEnd.ptr()[chainColPtr - 1] : 0;

      int64_t nRHS64 = nRHS;
      dispatchKernel(
          sym.commandQueue, pipeline,
          ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowsTillEnd.buffer()
                        offset:chainColPtr * sizeof(int64_t)
                       atIndex:0];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devChainRowSpan.buffer()
                        offset:chainColPtr * sizeof(int64_t)
                       atIndex:1];
            [encoder setBuffer:(__bridge id<MTLBuffer>)sym.devSpanStart.buffer() offset:0 atIndex:2];
            [encoder setBuffer:cBuffer offset:cOffset atIndex:3];
            [encoder setBytes:&numColItems length:sizeof(int64_t) atIndex:4];
            [encoder setBuffer:(__bridge id<MTLBuffer>)tempVecBuffer.buffer() offset:0 atIndex:5];
            [encoder setBytes:&ldc length:sizeof(int64_t) atIndex:6];
            [encoder setBytes:&nRHS64 length:sizeof(int64_t) atIndex:7];
            [encoder setBytes:&startRow length:sizeof(int64_t) atIndex:8];
          },
          (NSUInteger)numColItems);
    }
  }

  MetalSymbolicCtx& sym;
  int nRHS;
  MetalMirror<float> tempVecBuffer;
};

// Batched numeric context for float
template <>
struct MetalNumericCtx<std::vector<float*>> : NumericCtx<std::vector<float*>> {
  MetalNumericCtx(MetalSymbolicCtx& sym_, int64_t tempBufSize, int64_t numSpans, int batchSize_)
      : sym(sym_), numSpans_(numSpans), batchSize(batchSize_) {
    tempBuffer.resizeToAtLeast(tempBufSize * batchSize);
  }

  virtual ~MetalNumericCtx() override {}

  virtual void pseudoFactorSpans(std::vector<float*>* data, int64_t spanBegin,
                                 int64_t spanEnd) override {
    throw std::runtime_error(
        "MetalNumericCtx<std::vector<float*>>::pseudoFactorSpans not yet implemented");
  }

  virtual void doElimination(const SymElimCtx& elimData, std::vector<float*>* data,
                             int64_t lumpsBegin, int64_t lumpsEnd) override {
    throw std::runtime_error(
        "MetalNumericCtx<std::vector<float*>>::doElimination not yet implemented");
  }

  virtual void potrf(int64_t n, std::vector<float*>* data, int64_t offA) override {
    throw std::runtime_error("MetalNumericCtx<std::vector<float*>>::potrf not yet implemented");
  }

  virtual void trsm(int64_t n, int64_t k, std::vector<float*>* data, int64_t offA,
                    int64_t offB) override {
    throw std::runtime_error("MetalNumericCtx<std::vector<float*>>::trsm not yet implemented");
  }

  virtual void saveSyrkGemm(int64_t m, int64_t n, int64_t k, const std::vector<float*>* data,
                            int64_t offset) override {
    throw std::runtime_error(
        "MetalNumericCtx<std::vector<float*>>::saveSyrkGemm not yet implemented");
  }

  virtual void prepareAssemble(int64_t targetLump) override {}

  virtual void assemble(std::vector<float*>* data, int64_t rectRowBegin, int64_t dstStride,
                        int64_t srcColDataOffset, int64_t srcRectWidth, int64_t numBlockRows,
                        int64_t numBlockCols) override {
    throw std::runtime_error("MetalNumericCtx<std::vector<float*>>::assemble not yet implemented");
  }

  MetalSymbolicCtx& sym;
  int64_t numSpans_;
  int batchSize;
  MetalMirror<float> tempBuffer;
};

// Batched solve context for float
template <>
struct MetalSolveCtx<std::vector<float*>> : SolveCtx<std::vector<float*>> {
  MetalSolveCtx(MetalSymbolicCtx& sym_, int nRHS_, int batchSize_)
      : sym(sym_), nRHS(nRHS_), batchSize(batchSize_) {}

  virtual ~MetalSolveCtx() override {}

  virtual void sparseElimSolveL(const SymElimCtx& elimData, const std::vector<float*>* data,
                                int64_t lumpsBegin, int64_t lumpsEnd, std::vector<float*>* C,
                                int64_t ldc) override {
    throw std::runtime_error(
        "MetalSolveCtx<std::vector<float*>>::sparseElimSolveL not yet implemented");
  }

  virtual void sparseElimSolveLt(const SymElimCtx& elimData, const std::vector<float*>* data,
                                 int64_t lumpsBegin, int64_t lumpsEnd, std::vector<float*>* C,
                                 int64_t ldc) override {
    throw std::runtime_error(
        "MetalSolveCtx<std::vector<float*>>::sparseElimSolveLt not yet implemented");
  }

  virtual void symm(const std::vector<float*>* data, int64_t offset, int64_t n,
                    const std::vector<float*>* C, int64_t offC, int64_t ldc, std::vector<float*>* D,
                    int64_t ldd, float alpha) override {
    throw std::runtime_error("MetalSolveCtx<std::vector<float*>>::symm not yet implemented");
  }

  virtual void solveL(const std::vector<float*>* data, int64_t offset, int64_t n,
                      std::vector<float*>* C, int64_t offC, int64_t ldc) override {
    throw std::runtime_error("MetalSolveCtx<std::vector<float*>>::solveL not yet implemented");
  }

  virtual void gemv(const std::vector<float*>* data, int64_t offset, int64_t nRows, int64_t nCols,
                    const std::vector<float*>* A, int64_t offA, int64_t lda, float alpha) override {
    throw std::runtime_error("MetalSolveCtx<std::vector<float*>>::gemv not yet implemented");
  }

  virtual void assembleVec(int64_t chainColPtr, int64_t numColItems, std::vector<float*>* C,
                           int64_t ldc) override {
    throw std::runtime_error("MetalSolveCtx<std::vector<float*>>::assembleVec not yet implemented");
  }

  virtual void solveLt(const std::vector<float*>* data, int64_t offset, int64_t n,
                       std::vector<float*>* C, int64_t offC, int64_t ldc) override {
    throw std::runtime_error("MetalSolveCtx<std::vector<float*>>::solveLt not yet implemented");
  }

  virtual void gemvT(const std::vector<float*>* data, int64_t offset, int64_t nRows, int64_t nCols,
                     std::vector<float*>* A, int64_t offA, int64_t lda, float alpha) override {
    throw std::runtime_error("MetalSolveCtx<std::vector<float*>>::gemvT not yet implemented");
  }

  virtual void assembleVecT(const std::vector<float*>* C, int64_t ldc, int64_t chainColPtr,
                            int64_t numColItems) override {
    throw std::runtime_error("MetalSolveCtx<std::vector<float*>>::assembleVecT not yet implemented");
  }

  MetalSymbolicCtx& sym;
  int nRHS;
  int batchSize;
};

// Factory method implementations
NumericCtxBase* MetalSymbolicCtx::createNumericCtxForType(type_index tIdx, int64_t tempBufSize,
                                                          int batchSize) {
  int64_t numSpans = skel.spanStart.size() - 1;

  // Check for double precision - not supported on Metal
  if (tIdx == type_index(typeid(double)) || tIdx == type_index(typeid(std::vector<double*>))) {
    throw std::runtime_error(
        "Metal backend does not support double precision. "
        "Apple Silicon GPUs lack double-precision floating point support. "
        "Please use float or select a different backend (BackendFast or BackendCuda).");
  }

  if (tIdx == type_index(typeid(float))) {
    return new MetalNumericCtx<float>(*this, tempBufSize, numSpans);
  }
  if (tIdx == type_index(typeid(std::vector<float*>))) {
    return new MetalNumericCtx<std::vector<float*>>(*this, tempBufSize, numSpans, batchSize);
  }

  throw std::runtime_error("MetalSymbolicCtx: unsupported numeric type");
}

SolveCtxBase* MetalSymbolicCtx::createSolveCtxForType(type_index tIdx, int nRHS, int batchSize) {
  // Check for double precision - not supported on Metal
  if (tIdx == type_index(typeid(double)) || tIdx == type_index(typeid(std::vector<double*>))) {
    throw std::runtime_error(
        "Metal backend does not support double precision. "
        "Apple Silicon GPUs lack double-precision floating point support. "
        "Please use float or select a different backend (BackendFast or BackendCuda).");
  }

  if (tIdx == type_index(typeid(float))) {
    return new MetalSolveCtx<float>(*this, nRHS);
  }
  if (tIdx == type_index(typeid(std::vector<float*>))) {
    return new MetalSolveCtx<std::vector<float*>>(*this, nRHS, batchSize);
  }

  throw std::runtime_error("MetalSymbolicCtx: unsupported solve type");
}

// Public factory function
OpsPtr metalOps() { return OpsPtr(new MetalOps); }

}  // end namespace BaSpaCho
