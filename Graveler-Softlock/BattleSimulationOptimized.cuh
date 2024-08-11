#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <MemoryManagement.h>

__global__ void SimulateBattleOptimized(int* turns, int* paralysisCounts);

int SimulateBattlesOptimized(int iterations, int turns);