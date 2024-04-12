
struct Color {
	double r, g, b, a;
}

class RGBImage {
public:

	RGBImage(size_t h, size_t w);
	RGBImage(RGBImage& image);

	size_t height() const;
	size_t width() const;

	void resize(size_t h, size_t w);
	void fill(Color color);
	void randFill();

	void setPixel(size_t y, size_t x, struct Color);
	const Color& getPixel(size_t y, size_t x) const;

	void saveToFile(const std::string& path) const;

private:
	std::unique_ptr<std::vector<std::vector<struct Color>>> data;

	size_t m_Height;
	size_t m_Width;
};

using Image = RGBImage<double>;

RGBImage::RGBImage(size_t h, size_t w) : m_Width(w), m_Height(h) {

}

void RGBImage::resize(size_t h, size_t w) {
}

void RGBImage::fill(Color& color) {

}

void RGBImage<T>::randFill() {
	std::random_device rd;
	std::mt19937 gen{ rd()};
	std::uniform_real_distribution<> distrib(0.0, 1.0);
	for (size_t i = 0; i < m_Height; i++) {
		for (size_t j = 0; j < m_Width; j++) {
			m_RedChannel.at(i).at(j) = static_cast<T> (distrib(gen));
			m_GreenChannel.at(i).at(j) = static_cast<T> (distrib(gen));
			m_BlueChannel.at(i).at(j) = static_cast<T> (distrib(gen));
		}
	}
}

void RGBImage<T>::setPixel(size_t y, size_t x, Color& color) {
	m_RedChannel.at(y).at(x) = color.x()*255.0;
	m_GreenChannel.at(y).at(x) = color.y()*255.0;
	m_BlueChannel.at(y).at(x) = color.z()*255.0;
}

inline size_t RGBImage<T>::width() const {
	return m_Width;
}

inline size_t RGBImage<T>::height() const {
	return m_Height;
}

void RGBImage<T>::saveToFile(const std::string& path) const {
	std::ofstream os{ path, std::ios::out | std::ios::trunc};
	os << "P3\n" << m_Width << ' ' << m_Height << "\n255\n";
	for (size_t i{0}; i < m_Height; i++) {
		for (size_t j{0}; j < m_Width; j++) {
			os << static_cast<int> (m_RedChannel.at(i).at(j)) << ' '
					<< static_cast<int> (m_GreenChannel.at(i).at(j)) << ' '
					<< static_cast<int> (m_BlueChannel.at(i).at(j)) << '\n';
		}
	}
	os.close();
}

std::ostream& operator<<(std::ostream& os, const RGBImage& image) {
	for (int i=0; i < image.height(); i++) {
		for (int j=0; j < image.width(); j++) {
			os << "[" << image.getRedChannelPixel(j, i) <<
					", " << image.getGreenChannelPixel(j, i) <<
					", " << image.getBlueChannelPixel(j, i) << "]";
		}
		os << '\n';
	}
	return os;
}

