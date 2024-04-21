
#include <iostream>

#include "Image.h"

int main () {

	Image image1 { 10, 20 };
	image1.fill(Color{ 1.0, 1.0, 1.0, 1.0 });
	image1.saveToFile("images/image0.ppm");
	image1.randFill();
	image1.saveToFile("images/image1.ppm");
	Image image2 { image1 };
	image2.saveToFile("images/image2.ppm");
	Image image3 { std::move(image2) };
	image3.saveToFile("images/image3.ppm");
	Image image4;
	image4 = image1;
	image4.saveToFile("images/image4.ppm");
	Image image5;
	image5 = std::move(image1);
	image5.saveToFile("images/image5.ppm");

	size_t h = 100, w = 200;
	Image image6 { h, w };
	for (int i=0; i<h; i++) {
		for (int j=0; j<w; j++) {
			image6.setPixel(i, j, Color{0.5, 0.5, 0.5, 1.0});
		}
	}
	image6.saveToFile("images/image6.ppm");

	Image image7{ 300, 500 };
	image7.randFill();
	image7.saveToFile("images/image7.ppm");



	return 0;
}


