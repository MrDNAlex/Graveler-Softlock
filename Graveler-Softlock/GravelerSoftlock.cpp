#include "BattleSimulation.cuh"
#include "BattleSimulationOptimized.cuh"
#include <chrono>


int GetMaxParalysis(int iterations, int numOfTurns, int numOfPossibleMoves)
{
	auto before = std::chrono::high_resolution_clock::now();

	int* moveRolls = SimulateBattles(iterations, numOfTurns, numOfPossibleMoves);

	auto after = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(after - before);

	int paralysisCount = 0;

	for (int i = 0; i < iterations; i++)
	{
		if (moveRolls[i] > paralysisCount)
			paralysisCount = moveRolls[i];
	}

	//Free Memory
	delete[] moveRolls;

	printf("Time taken: %f seconds\n", duration.count());

	return paralysisCount;
}

int GetMaxParalysisOptimized(int iterations, int numOfTurns)
{
	auto before = std::chrono::high_resolution_clock::now();

	int paralysisCount = SimulateBattlesOptimized(iterations, numOfTurns);

	auto after = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(after - before);

	//int paralysisCount = 0;

	//for (int i = 0; i < iterations; i++)
	//{
	//	if (moveRolls[i] > paralysisCount)
	//		paralysisCount = moveRolls[i];
	//}

	////Free Memory
	//delete[] moveRolls;

	printf("Time taken: %f seconds\n", duration.count());

	return paralysisCount;
}


int main()
{
	int iterations = 10000000;
	int numOfPossibleMoves = 4;
	int numOfTurns = 231;

	printf("Max Paralysis Count Reached %d times\n", GetMaxParalysis(iterations, numOfTurns, numOfPossibleMoves));
	printf("Max Paralysis Count Reached %d times\n", GetMaxParalysisOptimized(iterations, numOfTurns));

	return 0;
}