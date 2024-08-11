#include "BattleSimulationOptimized.cuh"

__global__ void SimulateBattleOptimized(int* paralysisCounts)
{
	//Calculate GPU Core Index
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	//Skip if GPU core Index is greater than the number of iterations
	if (index >= iterations)
		return;

	//Initialize RNG Number Generator
	curandState state;
	curand_init(seed, index, 0, &state);

	//Initialize Move Count
	int paralysisCount = 0;

	//Loop through the number of turns
	for (int i = 0; i < turns; i++)
	{
		//Generate Random Number to determine if a move will cause paralysis
		int paralysisOdd = curand(&state) % 4;

		//Check if the move causes paralysis
		if (paralysisOdd == 0)
			paralysisCount++;
	}

	//Assign the Maximum Paralysis Count if it's greater than the current value
	atomicMax(paralysisCounts, paralysisCount);
}


int SimulateBattlesOptimized(int iterations)
{
	//Initialize GPU Variables
	int* gpuParalysisCount = 0;

	//Initialize CUDA Status
	cudaError_t cudaStatus;

	//Get the GPU Device
	cudaStatus = cudaSetDevice(0);

	//Assign Variables and Memory Space to the GPU
	cudaStatus = AssignMemory((void**)&gpuParalysisCount, sizeof(int));

	//Calculate the number of blocks and threads
	int threads = 1024;
	int blocks = (iterations + threads - 1) / threads; //976563

	//Run the Simulation on the GPU
	SimulateBattleOptimized << <blocks, threads >> > (gpuParalysisCount);

	//Synchronize the GPU (Wait for calculations to finish)
	cudaStatus = cudaDeviceSynchronize();

	//Initialize the Move Rolls Array
	int* moveRolls = new int[1];

	//Retreive the Move Rolls from the GPU
	cudaStatus = GetVariable(moveRolls, gpuParalysisCount, sizeof(int));

	//Free Up the GPU Memory
	cudaFree(gpuParalysisCount);

	return moveRolls[0];
}