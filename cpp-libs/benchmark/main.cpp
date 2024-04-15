
#include <iostream>
#include <chrono>
#include <string>
#include <memory>
#include <array>

#include "Benchmark.h"

int main () {

	const size_t size = 1000000;

	{
		Benchmark bench2("int[]", true);
		int arr[size];
		for (auto i=0; i<size; i++)
			arr[i] = 10;
	}

	{
		Benchmark bench("array<int> -- iterator", true);
		auto ptr = std::make_unique<std::array<int, size>>();
		int i = 0;
		for (auto it=ptr->begin(); it!=ptr->end(); ++it, i++)
			*it = i;
	}

	{
		Benchmark bench("array<int> -- index", true);
		auto ptr = std::make_unique<std::array<int, size>>();
		for (int i=0; i<ptr->size(); i++)
			(*ptr)[i] = i;
	}


	return 0;
}

