#include "BattleSimulation.cuh"
#include "BattleSimulationOptimized.cuh"
#include <chrono>


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
	//int iterations = 1000000000;
	int iterations = 100000000;
	int numOfPossibleMoves = 4;
	int numOfTurns = 231;
	unsigned long long RNGSeed = 12346752378;

	std::cout << "Enter the Number of Iterations: ";
	std::cin >> iterations; 

	std::cout << "Enter the Number of Possible Moves: ";
	std::cin >> numOfPossibleMoves;

	std::cout << "Enter the Number of Turns in the Battle: ";
	std::cin >> numOfTurns;

	std::cout << "Enter the RNG Seed: ";
	std::cin >> RNGSeed;

	GetMaxParalysis(iterations, numOfTurns, numOfPossibleMoves, RNGSeed);
	GetMaxParalysisOptimized(iterations);


	return 0;
}