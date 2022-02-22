#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(const float* A, const float* B, float* C,
                          int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements) {
    C[i] = A[i] + B[i];
  }
}

int vector_add(const float* h_A, const float* h_B, float* h_C,
               int numElements) {
  cudaError_t err = cudaSuccess;

  size_t size = numElements * sizeof(float);

  float* d_A = NULL;
  err = cudaMalloc((void**)&d_A, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Falied to allocate device vector A (error code: %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  float* d_B = NULL;
  err = cudaMalloc((void**)&d_B, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Falied to allocate device vector B (error code: %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  float* d_C = NULL;
  err = cudaMalloc((void**)&d_C, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Falied to allocate device vector C (error code: %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  printf("Copy input data from the host memory to the CUDA device\n");
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "Falied to copy vector A from host to device (error code: %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "Falied to copy vector B from host to device (error code: %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);

  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Falied to launch vectorAdd kernel (error code: %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  printf("Copy output data from the CUDA device to the host memory\n");
  err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "Falied to copy vector C from device to host (error code: %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_A);
  if (err != cudaSuccess) {
    fprintf(stderr, "Falied to free device vector A (error code: %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_B);
  if (err != cudaSuccess) {
    fprintf(stderr, "Falied to free device vector B (error code: %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_C);
  if (err != cudaSuccess) {
    fprintf(stderr, "Falied to free device vector C (error code: %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}