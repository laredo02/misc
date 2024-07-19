
#include <iostream>
#include <vector>
#include <complex>

using std::complex;
using std::vector;

int mandelbrot(complex<double> c, int max_iter) {
	complex<double> z { 0.0, 0.0 };
	for (int i=0; i<max_iter; i++) {
		if (std::abs(z)>2)
			return i;
		z = z*z + c;
	}
	return max_iter;
}

int main () {

	std::vector<int> = ;

	std::cout << mandelbrot({0.0, 1.0}, 1000) << std::endl;

	return 0;
}



