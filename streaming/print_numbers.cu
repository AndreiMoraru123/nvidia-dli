#include <stdio.h>
#include <unistd.h>

__global__ void printNumber(int number) { printf("%d\n", number); }

int main() {
  cudaStream_t streams[5];

  for (int i = 0; i < 5; ++i) {
    cudaStreamCreate(&streams[i]);
    printNumber<<<1, 1, 0, streams[i]>>>(i);
    cudaStreamDestroy(streams[i]);
  }
  cudaDeviceSynchronize();
}
