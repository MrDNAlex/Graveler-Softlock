#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <MemoryManagement.h>

__device__ int seed = 123423784;
__device__ int turns = 231;
__device__ int iterations = 1000000000;

/// <summary>
/// Simulates a Single Battle on the GPU
/// </summary>
/// <param name="paralysisCounts"> The Maximum number of Paralyses Counted</param>
/// <returns></returns>
__global__ void SimulateBattleOptimized( int* paralysisCounts);

/// <summary>
/// Simulates all the Battles on the GPU
/// </summary>
/// <param name="iterations"> The Number of Battles to Simulate</param>
/// <returns> Returns the Max Paralysis Count</returns>
int SimulateBattlesOptimized(int iterations);