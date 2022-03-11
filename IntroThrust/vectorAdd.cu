#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cstdio>

__global__ void vectorAdd(const float* A, const float* B, float* C,
                          int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements) {
    C[i] = A[i] + B[i];
  }
}

int main(void) {
  cudaError_t err = cudaSuccess;

  int numElements = 50000;

  printf("[Vector addition of %d elements]\n", numElements);

  thrust::host_vector<float> h_A(numElements);
  thrust::host_vector<float> h_B(numElements);

  const auto generator = [] { return rand() / (float)RAND_MAX; };
  std::generate(h_A.begin(), h_A.end(), generator);
  std::generate(h_B.begin(), h_B.end(), generator);

  thrust::device_vector<float> d_A(h_A);
  thrust::device_vector<float> d_B(h_B);

  thrust::device_vector<float> d_C(numElements);

  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);

  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(
      raw_pointer_cast(&d_A[0]), raw_pointer_cast(&d_B[0]),
      raw_pointer_cast(&d_C[0]), numElements);
  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Falied to launch vectorAdd kernel (error code: %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  printf("Copy output data from the CUDA device to the host memory\n");
  thrust::host_vector<float> h_C(d_C);

  for (int i = 0; i < numElements; i++) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d : %f != %f !\n",
              i, h_A[i] + h_B[i], h_C[i]);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test passed!\n");

  printf("Done.\n");

  return EXIT_SUCCESS;
}