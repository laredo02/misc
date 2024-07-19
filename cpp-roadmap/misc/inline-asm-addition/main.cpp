
#include <iostream>

int main() {
	int a = 10, b = 20, result;

	__asm__(
		"add %1, %2\n"
		"mov %2, %0\n"
		: "=r" (result)
		: "r" (a), "r" (b)
	);

	std::cout << result << std::endl;

	return 0;
}





