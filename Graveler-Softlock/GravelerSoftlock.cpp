#include "BattleSimulation.cuh"
#include <chrono>

/// <summary>
/// Simulates Battles on the GPU and Determines the Max number of Paralysis Rolls
/// </summary>
/// <param name="iterations"> The Number of Battles to Simulate </param>
/// <param name="numOfTurns"> Number of Turns in a Battle </param>
/// <param name="numOfPossibleMoves"> The Number of possible moves or turns in a Turn </param>
/// <param name="rngSeed"> The Seed for the RNG Numbers </param>
void GetMaxParalysis(int iterations, int numOfTurns, int numOfPossibleMoves, unsigned long long rngSeed)
{
	//Start the Clock!
	auto before = std::chrono::high_resolution_clock::now();

	//Simulate Battles on GPU
	int moveRolls = SimulateBattles(iterations, numOfTurns, numOfPossibleMoves, rngSeed);

	//Stop the Clock!
	auto after = std::chrono::high_resolution_clock::now();

	//Calculate the time taken
	auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(after - before);

	//Display the time taken and Max Number of Rolls
	printf("Max Paralysis Count in %f seconds: %d times\n", duration.count(), moveRolls);
}

/// <summary>
/// Determines if there are Active GPUs on the Computer
/// </summary>
/// <returns> True if GPUs are Detected, False otherwise </returns>
bool IsGPUActive()
{
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		std::cout << "Device Number: " << i << std::endl;
		std::cout << "  Device name: " << prop.name << std::endl;
		std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
		std::cout << "  Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
		std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
		std::cout << "  Max block dimensions: [" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]" << std::endl;
		std::cout << "  Max grid dimensions: [" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]" << std::endl;

		printf("\n");
	}

	if (nDevices == 0)
	{
		std::cout << "No CUDA devices found!" << std::endl;
		return false;
	}
	else
		return true;
}


int main()
{
	if (!IsGPUActive())
	{
		std::cout << "GPU not Detected, Program will not function without one" << std::endl;
		return 1;
	}

	//int iterations = 1000000000;
	int iterations = 0;
	int numOfPossibleMoves = 0;
	int numOfTurns = 0;
	unsigned long long RNGSeed = 0;

	std::cout << "Enter the Number of Iterations: ";
	std::cin >> iterations; 

	std::cout << "Enter the Number of Possible Moves: ";
	std::cin >> numOfPossibleMoves;

	std::cout << "Enter the Number of Turns in the Battle: ";
	std::cin >> numOfTurns;

	std::cout << "Enter the RNG Seed: ";
	std::cin >> RNGSeed;

	GetMaxParalysis(iterations, numOfTurns, numOfPossibleMoves, RNGSeed);

	std::cout << "Press Enter to Exit" << std::endl;

	return 0;
}