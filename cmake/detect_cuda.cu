/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdio.h>

int main(int argc, char** argv) {
  cudaDeviceProp dP;
  float min_cc = 6.0;

  int rc = cudaGetDeviceProperties(&dP, 0);
  if (rc != cudaSuccess) {
    cudaError_t error = cudaGetLastError();
    printf("CUDA error: %s", cudaGetErrorString(error));
    return rc; /* Failure */
  }
  if ((dP.major + (dP.minor / 10)) < min_cc) {
    printf("Min Compute Capability of %2.1f required:  %d.%d found", min_cc, dP.major, dP.minor);
    return 1; /* Failure */
  } else {
    printf("%d%d", dP.major, dP.minor);
    return 0; /* Success */
  }
}
