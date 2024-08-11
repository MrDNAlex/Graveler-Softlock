#include "BattleSimulation.cuh"

__global__ void SimulateBattle(int* turns, int* possibilities, int* iterations, int* moveRolls)
{
	//Calculate GPU Core Index
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	//Initialize RNG Number Generator
	curandState state;
	curand_init(1234, index, 0, &state);

	//Dereference Variables
	int numOfTurns = *turns;
	int numOfPossibilities = *possibilities;
	int numOfIterations = *iterations;

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

	moveRolls[index] = paralysisCount;
}

int* SimulateBattles(int iterations, int turns, int possibilities)
{
	//Initialize GPU Variables
	int* gpuTurns = 0;
	int* gpuMoveRolls = 0;
	int* gpuPossibilities = 0;
	int* gpuInterations = 0;

	//Initialize CUDA Status
	cudaError_t cudaStatus;

	//Get the GPU Device
	cudaStatus = cudaSetDevice(0);

	//Assign Variables and Memory Space to the GPU
	cudaStatus = AssignVariable((void**)&gpuTurns, &turns, sizeof(int));
	cudaStatus = AssignVariable((void**)&gpuPossibilities, &possibilities, sizeof(int));
	cudaStatus = AssignVariable((void**)&gpuInterations, &iterations, sizeof(int));
	cudaStatus = AssignMemory((void**)&gpuMoveRolls, sizeof(int), iterations);

	//Calculate the number of blocks and threads
	int threads = 1024;
	int blocks = (iterations + threads - 1) / threads;

	//Run the Simulation on the GPU
	SimulateBattle << <blocks, threads >> > (gpuTurns, gpuPossibilities, gpuInterations, gpuMoveRolls);

	//Synchronize the GPU (Wait for calculations to finish)
	cudaStatus = cudaDeviceSynchronize();

	//Initialize the Move Rolls Array
	int* moveRolls = new int[iterations];
	
	//Retreive the Move Rolls from the GPU
	cudaStatus = GetVariable(moveRolls, gpuMoveRolls, sizeof(int), iterations);

	//Free Up the GPU Memory
	cudaFree(gpuTurns);
	cudaFree(gpuPossibilities);
	cudaFree(gpuMoveRolls);
	cudaFree(gpuInterations);

	return moveRolls;
}