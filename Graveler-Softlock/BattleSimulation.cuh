#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <MemoryManagement.h>

/// <summary>
/// Simulates a Single Battle on the GPU
/// </summary>
/// <param name="turns"> The Number of Turns in the battle </param>
/// <param name="possibilities"> The Number of Possibilities to Occur in a Turn </param>
/// <param name="iterations"> The Number of iterations in the simulation </param>
/// <param name="paralysisCounts"> The Array of Number of Paralysis Counts </param>
/// <returns></returns>
//__global__ void SimulateBattle(int* turns, int* possibilities, int* iterations, int* paralysisCounts, unsigned long long* rngSeed);

__global__ void SimulateBattle(unsigned long long* simulationCount, bool* kill, int* maxParalysisCounts, unsigned long long* rngSeed);


/// <summary>
/// Simulates all the Battles on the GPU
/// </summary>
/// <param name="iterations"> The Number of Battles to Simulate</param>
/// <param name="turns"> The number of Turns that occur in the Battle </param>
/// <param name="possibilities"> The Number of possibilities for each turn (Basically the inverse of the probability)</param>
/// <param name="rngSeed"> The Seed for the Random Number Generator </param>
/// <returns> Returns the Paralysis Counts </returns>
int SimulateBattles(unsigned long long rngSeed);