
#include <fstream>
#include <vector>
#include <random>
#include <cassert>

#include <SDL2/SDL.h>

#define ASSERTIONS 1
#define BOUNDS_CHECKING 1
#define ECHO_CONSTRUCTION 1

using namespace std;

struct Color {
	double red, green, blue, alpha;
};

class Image {
public:
	Image();
	Image(size_t h, size_t w);
	Image(const Image& image);
	Image(Image&& image) noexcept;
	Image& operator=(const Image& image);
	Image& operator=(Image&& image) noexcept;
	~Image();

	size_t height() const;
	size_t width() const;

	void resize(size_t h, size_t w);
	void fill(const Color& color);
	void randFill();

	void setPixel(size_t y, size_t x, const Color& color);
	const Color& getPixel(size_t y, size_t x) const;

	void saveToFile(const std::string& path) const;

private:

	Uint32* m_Data;

	size_t m_Height;
	size_t m_Width;

};

Image::Image() : m_Height(0), m_Width(0), m_Data(nullptr) {
#if ECHO_CONSTRUCTION == 1
	cout << "Image::Image()" << endl;
#endif
}

Image::Image(size_t h, size_t w) : m_Width(w), m_Height(h) {
#if ECHO_CONSTRUCTION == 1
	cout << "Image::Image(size_t, size_t)" << endl;
#endif
	m_Data = new Color[m_Height*m_Width];
}

Image::Image(const Image& other) : m_Height(other.m_Height), m_Width(other.m_Width) {
#if ECHO_CONSTRUCTION == 1
	cout << "Image::Image(const Image&)" << endl;
#endif
	m_Data = new Color[m_Height*m_Width];
	for (int i=0; i<m_Height; i++) {
		for (int j=0; j<m_Width; j++) {
#if BOUNDS_CHECKING == 1
			assert(i*m_Width + j < m_Height*m_Width);
			assert(0 <= i && i <= m_Height);
			assert(0 <= j && j <= m_Width);
#endif
			m_Data[i*m_Width + j] = other.m_Data[i*m_Width + j];
		}
}
}

Image::Image(Image&& other) noexcept : m_Height(other.m_Height), m_Width(other.m_Width), m_Data(other.m_Data) {
#if ECHO_CONSTRUCTION == 1
	cout << "Image::Image(const Image&&)" << endl;
#endif
	other.m_Height = 0;
	other.m_Width = 0;
	other.m_Data = nullptr;
}

Image& Image::operator=(const Image& other) {
#if ECHO_CONSTRUCTION == 1
	cout << "Image::operation=(const Image&)" << endl;
#endif
	if (&other != this) {
		delete[] m_Data;

		m_Height = other.m_Height;
		m_Width = other.m_Width;

		m_Data = new Color[m_Height*m_Width];
		for (int i=0; i<m_Height; i++) {
			for (int j=0; j<m_Width; j++) {
#if BOUNDS_CHECKING == 1
			assert(i*m_Width + j < m_Height*m_Width);
			assert(0 <= i && i <= m_Height);
			assert(0 <= j && j <= m_Width);
#endif
				m_Data[i*m_Width + j] = other.m_Data[i*m_Width + j];
			}
		}

	}
	return *this;
}

Image& Image::operator=(Image&& other) noexcept {
#if ECHO_CONSTRUCTION == 1
	cout << "Image::operation=(const Image&&)" << endl;
#endif
	if (&other != this) {
		delete[] m_Data;

		m_Data = other.m_Data;
		m_Height = other.m_Height;
		m_Width = other.m_Width;

		other.m_Data = nullptr;
		other.m_Height = 0;
		other.m_Width = 0;
	}
	return *this;
}

Image::~Image() {
#if ECHO_CONSTRUCTION == 1
	cout << "Image::~Image()" << endl;
#endif
	delete[] m_Data;
}

void Image::resize(size_t height, size_t width) {
	delete[] m_Data;
	m_Height = height;
	m_Width = width;
	m_Data = new Color[m_Height*m_Width];
}

void Image::fill(const Color& color) {
	for (int i = 0; i < m_Height; i++) {
		for (int j = 0; j < m_Width; j++) {
#if BOUNDS_CHECKING == 1
			assert(i*m_Width + j < m_Height*m_Width);
			assert(0 <= i && i <= m_Height);
			assert(0 <= j && j <= m_Width);
#endif
			m_Data[i*m_Width + j] = color;
		}
	}
}

void Image::randFill() {

	std::random_device rd;
	std::mt19937 gen{ rd()};
	std::uniform_real_distribution<> distrib(0.0, 1.0);

	for (int i = 0; i < m_Height; i++) {
		for (int j = 0; j < m_Width; j++) {
#if BOUNDS_CHECKING == 1
			assert(i*m_Width + j < m_Height*m_Width);
			assert(0 <= i && i <= m_Height);
			assert(0 <= j && j <= m_Width);
#endif
			m_Data[i*m_Width + j] = Color{ distrib(gen), distrib(gen), distrib(gen), 1.0 };
		}
	}

}

void Image::setPixel(size_t y, size_t x, const Color& color) {

#if BOUNDS_CHECKING == 1
	assert(y*m_Width + x < m_Height*m_Width);
	assert(0 <= y && y <= m_Height);
	assert(0 <= x && x <= m_Width);
#endif

#if ASSERTIONS == 1
	assert(0.0 <= color.red && color.red <= 1.0);
	assert(0.0 <= color.green && color.green <= 1.0);
	assert(0.0 <= color.blue && color.blue <= 1.0);
	assert(0.0 <= color.alpha && color.alpha <= 1.0);
#endif

	m_Data[y*m_Width + x] = color;

}

inline size_t Image::width() const { return m_Width; }
inline size_t Image::height() const { return m_Height; }

void Image::saveToFile(const std::string& path) const {
	std::ofstream os{ path, std::ios::out | std::ios::trunc};
	os << "P3\n" << m_Width << ' ' << m_Height << "\n255\n";
	for (int i=0; i < m_Height; i++) {
		for (int j=0; j < m_Width; j++) {
#if BOUNDS_CHECKING == 1
			assert(i*m_Width + j < m_Height*m_Width);
			assert(0 <= i && i <= m_Height);
			assert(0 <= j && j <= m_Width);
#endif
			int red = static_cast<int>(m_Data[i*m_Width + j].red*255.0);
			int green = static_cast<int>(m_Data[i*m_Width + j].green*255.0);
			int blue = static_cast<int>(m_Data[i*m_Width + j].blue*255.0);
#if ASSERTIONS == 1
			assert(0 <= red && red <= 255 && 0 <= green && green <= 255 && 0 <= blue && blue <= 255);
#endif
			os << red << ' ' << green << ' ' << blue << '\n';
		}
	}
	os.close();
}










