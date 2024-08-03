
#include <iostream>
#include <cassert>
#include <array>
#include <complex>

#include "Image.hpp"

using std::cout;
using std::complex;
using std::array;

std::array<char, 10> charlist { '@', '#', '%', '*', '+', ';', ':', ',', '.', ' ' };

int mandelbrot(complex<double> c, int max_iter) {
	complex<double> z { 0.0, 0.0 };
	for (int i=0; i<max_iter; i++) {
		if (std::abs(z)>2)
			return i;
		z = z*z + c;
	}
	return max_iter;
}

int map_interval(const int min_in, const int max_in, const int min_out, const int max_out, const int value) {
	double factor = value/(double)(max_in-min_in);
	return (int) min_out+(max_out-min_out)*factor;
}

int main () {

	const auto step = 0.01;

	const auto xi = -2.0;
	const auto xf = 2.0;

	const auto yi = 2.0;
	const auto yf = -2.0;

	const auto depth = 9;


	for (double y=yi; y>=yf; y-=step) {
		for (double x=xi; x<=xf; x+=step) {
			int m = mandelbrot(complex(x, y), depth);
			cout << charlist[m] << ' ';
		}
		cout << '\n';
	}

	return 0;
}

