
#pragma once

#include <ostream>
#include <cmath>

struct XYZ {

public:

	XYZ();
	XYZ(const double x, const double y, const double z);


	double x() const;
	double y() const;
	double z() const;
	double operator[](size_t i) const;
	double operator[](size_t i);


	double norm() const;
	double normSquared() const;
	void toUnit();
	XYZ unit() const;
	friend double dot(const XYZ& u, const XYZ& v);
	friend XYZ cross(const XYZ& u, const XYZ& v);

	
	XYZ operator-() const;

	XYZ& operator+=(const XYZ& xyz);
	XYZ& operator+=(const double value);

	XYZ& operator-=(const XYZ& xyz);
	XYZ& operator-=(const double value);

	XYZ& operator*=(const XYZ& xyz);
	XYZ& operator*=(const double value);

	XYZ& operator/=(const XYZ& xyz);
	XYZ& operator/=(const double value);

	friend XYZ operator+(const XYZ& a, const XYZ& b);
	friend XYZ operator+(const XYZ& a, const double b);
	friend XYZ operator+(const double a, const XYZ& b);

	friend XYZ operator-(const XYZ& a, const XYZ& b);
	friend XYZ operator-(const XYZ& a, const double b);
	friend XYZ operator-(const double a, const XYZ& b);

	friend XYZ operator*(const XYZ& a, const XYZ& b);
	friend XYZ operator*(const XYZ& a, const double b);
	friend XYZ operator*(const double a, const XYZ& b);

	friend XYZ operator/(const XYZ& a, const XYZ& b);
	friend XYZ operator/(const XYZ& a, const double b);


	XYZ& rotateX(const XYZ& center, const double alpha);
	XYZ& rotateY(const XYZ& center, const double beta);
	XYZ& rotateZ(const XYZ& center, const double gamma);

	XYZ& rotateXYZ(const XYZ& center, const double alpha, const double beta, const double gamma);
	XYZ& rotateAAxis(const XYZ& axisDir, const double theta);


	friend std::ostream& operator<<(std::ostream& out, const XYZ& xyz);

private:

	double m_xyz[3];

};

using Vector3 = XYZ;

















