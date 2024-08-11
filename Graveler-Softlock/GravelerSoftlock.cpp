#include "BattleSimulation.cuh"
#include "BattleSimulationOptimized.cuh"
#include <chrono>


void GetMaxParalysis(int iterations, int numOfTurns, int numOfPossibleMoves)
{
	//Start the Clock!
	auto before = std::chrono::high_resolution_clock::now();

	//Simulate Battles on GPU
	int* moveRolls = SimulateBattles(iterations, numOfTurns, numOfPossibleMoves);

	//Find the Max Number of Rolls
	int paralysisCount = 0;
	for (int i = 0; i < iterations; i++)
	{
		if (moveRolls[i] > paralysisCount)
			paralysisCount = moveRolls[i];
	}

	//Free Memory
	delete[] moveRolls;

	//Stop the Clock!
	auto after = std::chrono::high_resolution_clock::now();

	//Calculate the time taken
	auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(after - before);

	//Display the time taken and Max Number of Rolls
	printf("Max Paralysis Count in %f seconds: %d times\n", duration.count(), paralysisCount);
}

int GetMaxParalysisOptimized(int iterations)
{
	//Start the Clock!
	auto before = std::chrono::high_resolution_clock::now();

	//Simulate Battles on GPU
	int paralysisCount = SimulateBattlesOptimized(iterations);

	//Stop the Clock!
	auto after = std::chrono::high_resolution_clock::now();

	//Calculate the time taken
	auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(after - before);

	printf("Max Paralysis Count in %f seconds: %d times\n", duration.count(), paralysisCount);

	return paralysisCount;
}


int main()
{
	int iterations = 1000000000;
	int numOfPossibleMoves = 4;
	int numOfTurns = 231;

	//GetMaxParalysis(iterations, numOfTurns, numOfPossibleMoves);
	GetMaxParalysisOptimized(iterations);


	return 0;
}