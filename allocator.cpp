#include <chrono>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <ratio>
#include <sys/types.h>

extern "C" {

struct DeltaTimer {
  std::chrono::high_resolution_clock::time_point prev;

  DeltaTimer() : prev(std::chrono::high_resolution_clock::now()) {}

  double operator()() {
    auto now = std::chrono::high_resolution_clock::now();
    double delta =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - this->prev)
            .count();
    this->prev = now;
    return delta;
  }
};

DeltaTimer dt;

void *my_alloc(ssize_t size, int device, cudaStream_t stream) {
  double time = dt();
  std::ofstream results;
  results.open("allocations.csv", std::ios::out | std::ios::app);
  void *ptr;
  cudaMalloc(&ptr, size);
  results << ptr << ", " << size << ", " << time << std::endl;
  results.close();
  return ptr;
}

void my_free(void *ptr, ssize_t size, int device, cudaStream_t stream) {
  double time = dt();
  std::ofstream results;
  results.open("allocations.csv", std::ios::out | std::ios::app);
  results << ptr << ", "
          << "-" << size << ", " << time << std::endl;
  results.close();
  cudaFree(ptr);
}
}
