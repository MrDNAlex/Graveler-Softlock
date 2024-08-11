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

	//Skip if GPU core Index is greater than the number of iterations
	if (index >= numOfIterations)
		return;

	//Initialize Move Count
	int moveCounts[4] = { 0, 0, 0, 0 };

	//Initialize Shared Memory for Paralysis Counts in a Block
	__shared__ int counts[1024];

	//Create a Shared Random Number Generator between all threads
	__shared__ curandState sharedRNG;
	if (threadIdx.x == 0)
		curand_init(seed, index, 0, &sharedRNG);
	__syncthreads();

	//Loop through the number of turns
	for (int i = 0; i < numOfTurns; i = i + 16)
	{
		//Generate a Random Number
		unsigned int paralysisOdd = curand(&sharedRNG);

		//Extract 2 Bits from the Random Number
		unsigned char random1 = (paralysisOdd >> 0) & 0x03;   // First 2 bits (0-3)
		unsigned char random2 = (paralysisOdd >> 2) & 0x03;   // Next 2 bits (0-3)
		unsigned char random3 = (paralysisOdd >> 4) & 0x03;   // Next 2 bits (0-3)
		unsigned char random4 = (paralysisOdd >> 6) & 0x03;   // Next 2 bits (0-3)
		unsigned char random5 = (paralysisOdd >> 8) & 0x03;   // Next 2 bits (0-3)
		unsigned char random6 = (paralysisOdd >> 10) & 0x03;  // Next 2 bits (0-3)
		unsigned char random7 = (paralysisOdd >> 12) & 0x03;  // Next 2 bits (0-3)
		unsigned char random8 = (paralysisOdd >> 14) & 0x03;  // Next 2 bits (0-3)
		unsigned char random9 = (paralysisOdd >> 16) & 0x03;  // Next 2 bits (0-3)
		unsigned char random10 = (paralysisOdd >> 18) & 0x03; // Next 2 bits (0-3)
		unsigned char random11 = (paralysisOdd >> 20) & 0x03; // Next 2 bits (0-3)
		unsigned char random12 = (paralysisOdd >> 22) & 0x03; // Next 2 bits (0-3)
		unsigned char random13 = (paralysisOdd >> 24) & 0x03; // Next 2 bits (0-3)
		unsigned char random14 = (paralysisOdd >> 26) & 0x03; // Next 2 bits (0-3)
		unsigned char random15 = (paralysisOdd >> 28) & 0x03; // Next 2 bits (0-3)
		unsigned char random16 = (paralysisOdd >> 30) & 0x03; // Last 2 bits (0-3)

		//Add to the right Array Index for the Move used
		moveCounts[random1]++;
		moveCounts[random2]++;
		moveCounts[random3]++;
		moveCounts[random4]++;
		moveCounts[random5]++;
		moveCounts[random6]++;
		moveCounts[random7]++;
		moveCounts[random8]++;
		moveCounts[random9]++;
		moveCounts[random10]++;
		moveCounts[random11]++;
		moveCounts[random12]++;
		moveCounts[random13]++;
		moveCounts[random14]++;
		moveCounts[random15]++;
		moveCounts[random16]++;
	}

	//Add the Paralysis Count to the Shared Memory
	counts[threadIdx.x] = moveCounts[0];

	//Find the Maximum Paralysis Count in the Thread Group
	int threadMax = 0;
	if (threadIdx.x == 0)
	{
		int max = 0;
		for (int i = 0; i < 1024; i++)
		{
			if (counts[i] > max)
				max = counts[i];
		}

		threadMax = max;
	}

	//Synchronize the Threads
	__syncthreads();

	//Check and replace the Global Paralysis Count if it's the highest so far
	if (threadIdx.x == 0)
	{
		atomicMax(paralysisCounts, threadMax);
	}
}

int SimulateBattles(int iterations, int turns, int possibilities, unsigned long long rngSeed)
{
	//Initialize GPU Variables
	int* gpuTurns = 0;
	int* gpuMoveRolls = 0;
	int* gpuPossibilities = 0;
	int* gpuInterations = 0;
	unsigned long long* gpuRNGSeed = 0;
	curandState gpuRNG;

	//Initialize CUDA Status
	cudaError_t cudaStatus;

	//Get the GPU Device
	cudaStatus = cudaSetDevice(0);

	//Assign Variables and Memory Space to the GPU
	cudaStatus = AssignVariable((void**)&gpuTurns, &turns, sizeof(int));
	cudaStatus = AssignVariable((void**)&gpuPossibilities, &possibilities, sizeof(int));
	cudaStatus = AssignVariable((void**)&gpuInterations, &iterations, sizeof(int));
	cudaStatus = AssignVariable((void**)&gpuRNGSeed, &rngSeed, sizeof(unsigned long long));
	//cudaStatus = AssignVariable((void**)&gpuRNG, &state, sizeof(curandState));
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
	cudaFree(gpuRNGSeed);

	return moveRolls[0];
}