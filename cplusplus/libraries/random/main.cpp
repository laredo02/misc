
#include <iostream>
#include <random>

#include "Random.h"

int main () {

	std::cout << randomInt<int>(10, 20) << std::endl;
	std::cout << randomReal<double>(20.0, 30.0) << std::endl;

	return 0;
}

