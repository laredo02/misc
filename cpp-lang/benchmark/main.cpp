

#include <iostream>
#include <chrono>
#include <string>
#include <memory>
#include <array>

class Benchmark {
public:

	Benchmark(bool l) {
		name = "Benchmark";
		begin = std::chrono::steady_clock::now();
		laponly = l;
	}

	Benchmark(const std::string& n, bool l) : name(n) {
		begin = std::chrono::steady_clock::now();
		laponly = l;
	}

	Benchmark(const Benchmark& bench) = delete;
	Benchmark& operator=(const Benchmark& bench) = delete;

	~Benchmark() {
		if (laponly)
			std::cout << name << " " << lap() << "ms" << std::endl;
	}

	int64_t lap() {
		auto diff = std::chrono::steady_clock::now() - begin;
		return std::chrono::duration_cast<std::chrono::milliseconds>(diff).count();
	}

private:
	std::string name;
	std::chrono::steady_clock::time_point begin;
	bool laponly;
};

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

