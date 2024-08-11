#include "BattleSimulationOptimized.cuh"

__global__ void SimulateBattleOptimized(int* turns, int* paralysisCounts)
{
	//Calculate GPU Core Index
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	//Dereference Variables
	int numOfTurns = *turns;

	//Initialize Shared Memory
	//__shared__ int blockMax;

	//If it's the First Thread in the Block, Initialize the BlockMax
	//if (threadIdx.x == 0)
	//	blockMax = 0;

	

	//Skip if GPU core Index is greater than the number of iterations
	if (index >= 1000000000)
		return;

	//Initialize RNG Number Generator
	curandState state;
	curand_init(1234, index, 0, &state);

	//Initialize Move Count
	int paralysisCount = 0;

	//Loop through the number of turns
	for (int i = 0; i < numOfTurns; i++)
	{
		//Generate Random Number to determine if a move will cause paralysis
		int paralysisOdd = curand(&state) % 4;

		//Check if the move causes paralysis
		if (paralysisOdd == 0)
			paralysisCount++;
	}

	////Synchronize Threads
	//__syncthreads();

	//atomicMax(&blockMax, paralysisCount);
	//__syncthreads();

	//if (threadIdx.x == 0) {
	//	atomicMax(paralysisCounts, blockMax);
	//}

	atomicMax(paralysisCounts, paralysisCount);
}

/// <summary>
/// Simulates all the Battles on the GPU
/// </summary>
/// <param name="iterations"> The Number of Battles to Simulate</param>
/// <param name="turns"> The number of Turns that occur in the Battle </param>
/// <returns> Returns the Max Paralysis Count</returns>
int SimulateBattlesOptimized(int iterations, int turns)
{
	//Initialize GPU Variables
	int* gpuTurns = 0;
	int* gpuParalysisCount = 0;

	//Initialize CUDA Status
	cudaError_t cudaStatus;

	//Get the GPU Device
	cudaStatus = cudaSetDevice(0);

	//Assign Variables and Memory Space to the GPU
	cudaStatus = AssignVariable((void**)&gpuTurns, &turns, sizeof(int));
	//cudaStatus = AssignVariable((void**)&gpuInterations, &iterations, sizeof(int));
	cudaStatus = AssignMemory((void**)&gpuParalysisCount, sizeof(int));

	//Calculate the number of blocks and threads
	int threads = 1024;
	int blocks = (iterations + threads - 1) / threads;

	//Run the Simulation on the GPU
	SimulateBattleOptimized << <blocks, threads >> > (gpuTurns, gpuParalysisCount);

	//Synchronize the GPU (Wait for calculations to finish)
	cudaStatus = cudaDeviceSynchronize();

	//Initialize the Move Rolls Array
	//int* moveRolls = new int[iterations];
	int* moveRolls = new int[1];

	//Retreive the Move Rolls from the GPU
	cudaStatus = GetVariable(moveRolls, gpuParalysisCount, sizeof(int));

	//Free Up the GPU Memory
	cudaFree(gpuTurns);
	cudaFree(gpuParalysisCount);
	//cudaFree(gpuInterations);

	return moveRolls[0];
}