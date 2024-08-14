#include "BattleSimulation.cuh"

#define ROUNDS 1000000000
#define TURNS 231

__global__ void SimulateBattle(unsigned long long* simulationCount, int* maxParalysisCounts, unsigned long long* rngSeed)
{
	//Calculate GPU Core Index
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	//Skip if the Index is above the number of simulations
	if (index >= ROUNDS)
		return;

	//Initialize the RNG Generator
	curandState RNG;
	curand_init(*rngSeed, index, 0, &RNG);

	//Enter a While Loop to simulate Battles, it doesn't stop until all running cores finish all 1 Billion Simulations
	while (*simulationCount < ROUNDS - (index * 2))
	{
		int paralysisCount = 0;
		int turnCount = 0;

		//Simulate a Single Battle
		while (turnCount <= TURNS)
		{
			//Generate a 32 Bit Random Number
			unsigned int roll = curand(&RNG);

			//Extract 2 Bits from the Random Number and check if it is a Paralysis (x16)
			for (int i = 0; i < 16; i++)
			{
				int shift = i * 2;
				unsigned int currentRoll = (roll >> shift) & 0x03;
				if (currentRoll == 0)
					paralysisCount++;

				turnCount++;
			}
		}

		//Perform Atomic Max to get the Maximum Paralysis Count and Add 1 to the Simulations Count
		atomicMax(maxParalysisCounts, paralysisCount);
		atomicAdd(simulationCount, 1);
	}
}

int SimulateBattles(unsigned long long rngSeed)
{
	unsigned long long zeroLong = 0;
	int zero = 0;

	//Initialize GPU Variables
	int* gpuParalysisCount = 0;
	unsigned long long* gpuSimulationCount = 0;
	unsigned long long* gpuRNGSeed = 0;

	//Initialize CUDA Status
	cudaError_t cudaStatus;

	//Get the GPU Device
	cudaStatus = cudaSetDevice(0);

	//Assign Variables and Memory Space to the GPU
	cudaStatus = AssignVariable((void**)&gpuSimulationCount, &zeroLong, sizeof(unsigned long long));
	cudaStatus = AssignVariable((void**)&gpuRNGSeed, &rngSeed, sizeof(unsigned long long));
	cudaStatus = AssignVariable((void**)&gpuParalysisCount, &zero, sizeof(int));

	//Calculate the number of blocks and threads
	int threads = 1024;
	int blocks = 80;
	
	//Run the Simulation on the GPU
	SimulateBattle << <blocks, threads >> > (gpuSimulationCount, gpuParalysisCount, gpuRNGSeed);

	//Synchronize the GPU (Wait for calculations to finish)
	cudaStatus = cudaDeviceSynchronize();

	//Initialize the Move Rolls Array
	int* paralysisCount = new int[1];

	//Retreive the Move Rolls from the GPU
	cudaStatus = GetVariable(paralysisCount, gpuParalysisCount, sizeof(int));

	//Free Up the GPU Memory
	cudaFree(gpuParalysisCount);
	cudaFree(gpuRNGSeed);
	cudaFree(gpuSimulationCount);

	return paralysisCount[0];
}