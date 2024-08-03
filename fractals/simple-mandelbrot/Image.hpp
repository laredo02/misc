
#include <iostream>
#include <string>
#include <vector>

struct RGBA {
	double r;
	double g;
	double b;
	double a;
};

class Image {
public:
	Image(const int w, const int h);
	void setPixel(const int x, const int y, const RGBA& color);
	void writeToFile(std::string name) const;

private:
	int width;
	int height;
	std::vector<RGBA> data;
};

Image::Image(const int w, const int h) : width{w}, height{h} {}

void Image::setPixel(const int x, const int y, const RGBA& color) { data[width*x+y] = color; }

void Image::writeToFile(std::string name) const {
	
}





