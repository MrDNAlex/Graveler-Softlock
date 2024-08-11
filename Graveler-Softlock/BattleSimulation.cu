#include "BattleSimulation.cuh"

__global__ void SimulateBattle(int* turns, int* possibilities, int* iterations, int* paralysisCounts, unsigned long long* rngSeed)
{
	//Calculate GPU Core Index
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	//Dereference Variables
	int numOfTurns = *turns;
	int numOfPossibilities = *possibilities;
	int numOfIterations = *iterations;
	unsigned long long seed = *rngSeed;

	//Initialize RNG Number Generator
	curandState state;
	curand_init(seed, index, 0, &state);

	//Skip if GPU core Index is greater than the number of iterations
	if (index >= numOfIterations)
		return;

	//Initialize Move Count
	int paralysisCount = 0;

	//Loop through the number of turns
	for (int i = 0; i < numOfTurns; i++)
	{
		//Generate Random Number to determine if a move will cause paralysis
		int paralysisOdd = curand(&state) % numOfPossibilities;

		//Check if the move causes paralysis
		if (paralysisOdd == 0)
			paralysisCount++;
	}

	atomicMax(paralysisCounts, paralysisCount);
}

int SimulateBattles(int iterations, int turns, int possibilities, unsigned long long rngSeed)
{
	//Initialize GPU Variables
	int* gpuTurns = 0;
	int* gpuMoveRolls = 0;
	int* gpuPossibilities = 0;
	int* gpuInterations = 0;
	unsigned long long* gpuRNGSeed = 0;

	//Initialize CUDA Status
	cudaError_t cudaStatus;

	//Get the GPU Device
	cudaStatus = cudaSetDevice(0);

	//Assign Variables and Memory Space to the GPU
	cudaStatus = AssignVariable((void**)&gpuTurns, &turns, sizeof(int));
	cudaStatus = AssignVariable((void**)&gpuPossibilities, &possibilities, sizeof(int));
	cudaStatus = AssignVariable((void**)&gpuInterations, &iterations, sizeof(int));
	cudaStatus = AssignVariable((void**)&gpuRNGSeed, &rngSeed, sizeof(unsigned long long));
	cudaStatus = AssignMemory((void**)&gpuMoveRolls, sizeof(int));

	//Calculate the number of blocks and threads
	int threads = 1024;
	int blocks = (iterations + threads - 1) / threads;

	//Run the Simulation on the GPU
	SimulateBattle << <blocks, threads >> > (gpuTurns, gpuPossibilities, gpuInterations, gpuMoveRolls, gpuRNGSeed);

	//Synchronize the GPU (Wait for calculations to finish)
	cudaStatus = cudaDeviceSynchronize();

	//Initialize the Move Rolls Array
	int* moveRolls = new int[1];
	
	//Retreive the Move Rolls from the GPU
	cudaStatus = GetVariable(moveRolls, gpuMoveRolls, sizeof(int));

	//Free Up the GPU Memory
	cudaFree(gpuTurns);
	cudaFree(gpuPossibilities);
	cudaFree(gpuMoveRolls);
	cudaFree(gpuInterations);

	return moveRolls[0];
}