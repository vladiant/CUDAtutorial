
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "vectorAdd.hpp"

int main(int argc, char* argv[]) {
  int numElements = 50000;

  size_t size = numElements * sizeof(float);
  printf("[Vector addition of %d elements]\n", numElements);

  float* h_A = (float*)malloc(size);
  float* h_B = (float*)malloc(size);
  float* h_C = (float*)malloc(size);

  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    fprintf(stderr, "Falied to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < numElements; i++) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }

  int result = vector_add(h_A, h_B, h_C, numElements);
  if (result != EXIT_SUCCESS) {
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < numElements; i++) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d : %f != %f !\n",
              i, h_A[i] + h_B[i], h_C[i]);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test passed!\n");

  free(h_A);
  free(h_B);
  free(h_C);

  printf("Done.\n");

  return EXIT_SUCCESS;
}