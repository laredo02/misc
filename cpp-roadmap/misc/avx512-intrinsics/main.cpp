
#include <iostream>
#include <immintrin.h>

int main() {

	float arr1[16] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
	float arr2[16] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
	float arr3[16];

	__m512 a = _mm512_loadu_ps(arr1);
	__m512 b = _mm512_loadu_ps(arr2);
	__m512 c = _mm512_add_ps(a, b);
	_mm512_storeu_ps(arr3, c);
	
	for (float f: arr3)
		std::cout << f << std::endl;

	return 0;
}

