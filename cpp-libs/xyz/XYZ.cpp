
#include "XYZ.h"

#define DEG_TO_RAD(deg) ((deg)*(M_PI/180.0))

XYZ::XYZ() : m_xyz{ 0.0, 0.0, 0.0} {}
XYZ::XYZ(const double x, const double y, const double z) : m_xyz{x, y, z} {}

double XYZ::x() const { return m_xyz[0]; }
double XYZ::y() const { return m_xyz[1]; }
double XYZ::z() const { return m_xyz[2]; }

XYZ XYZ::operator-() const { return XYZ{ -m_xyz[0], -m_xyz[1], -m_xyz[2]}; }
double XYZ::operator[](size_t i) const { return m_xyz[i]; }
double XYZ::operator[](size_t i) { return m_xyz[i]; }

inline XYZ& XYZ::operator+=(const XYZ& xyz) {
	m_xyz[0] += xyz.m_xyz[0];
	m_xyz[1] += xyz.m_xyz[1];
	m_xyz[2] += xyz.m_xyz[2];
	return *this;
}

inline XYZ& XYZ::operator+=(const double value) {
	m_xyz[0] += value;
	m_xyz[1] += value;
	m_xyz[2] += value;
	return *this;
}

inline XYZ& XYZ::operator-=(const XYZ& xyz) {
	m_xyz[0] -= xyz.m_xyz[0];
	m_xyz[1] -= xyz.m_xyz[1];
	m_xyz[2] -= xyz.m_xyz[2];
	return *this;
}

inline XYZ& XYZ::operator-=(const double value) {
	m_xyz[0] -= value;
	m_xyz[1] -= value;
	m_xyz[2] -= value;
	return *this;
}

inline XYZ& XYZ::operator*=(const XYZ& xyz) {
	m_xyz[0] *= xyz.m_xyz[0];
	m_xyz[1] *= xyz.m_xyz[1];
	m_xyz[2] *= xyz.m_xyz[2];
	return *this;
}

inline XYZ& XYZ::operator*=(const double value) {
	m_xyz[0] *= value;
	m_xyz[1] *= value;
	m_xyz[2] *= value;
	return *this;
}

inline XYZ& XYZ::operator/=(const XYZ& xyz) {
	m_xyz[0] /= xyz.m_xyz[0];
	m_xyz[1] /= xyz.m_xyz[1];
	m_xyz[2] /= xyz.m_xyz[2];
	return *this;
}

inline XYZ& XYZ::operator/=(const double value) {
	m_xyz[0] /= value;
	m_xyz[1] /= value;
	m_xyz[2] /= value;
	return *this;
}

inline XYZ operator+(const XYZ& a, const XYZ& b) {
	return XYZ(a.m_xyz[0] + b.m_xyz[0], a.m_xyz[1] + b.m_xyz[1], a.m_xyz[2] + b.m_xyz[2]);
}

inline XYZ operator+(const XYZ& a, const double b) {
	return XYZ{ a.m_xyz[0] + b, a.m_xyz[1] + b, a.m_xyz[2] + b };
}

inline XYZ operator+(const double a, const XYZ& b) {
	return XYZ{ a + b.m_xyz[0], a + b.m_xyz[1], a + b.m_xyz[2] };
}

inline XYZ operator-(const XYZ& a, const XYZ& b) {
	return XYZ{ a.m_xyz[0] - b.m_xyz[0], a.m_xyz[1] - b.m_xyz[1], a.m_xyz[2] - b.m_xyz[2] };
}

inline XYZ operator-(const XYZ& a, const double b) {
	return XYZ(a.m_xyz[0] + b, a.m_xyz[1] + b, a.m_xyz[2] + b);
}

inline XYZ operator-(const double a, const XYZ& b) {
	return XYZ(a + b.m_xyz[0], a + b.m_xyz[1], a + b.m_xyz[2]);
}

inline XYZ operator*(const XYZ& a, const XYZ& b) {
	return XYZ(a.m_xyz[0] * b.m_xyz[0], a.m_xyz[1] * b.m_xyz[1], a.m_xyz[2] * b.m_xyz[2]);
}

inline XYZ operator*(const XYZ& a, double b) {
	return XYZ(a.m_xyz[0] * b, a.m_xyz[1] * b, a.m_xyz[2] * b);
}

inline XYZ operator*(const double a, const XYZ& b) {
	return XYZ(a * b.m_xyz[0], a * b.m_xyz[1], a * b.m_xyz[2]);
}

inline XYZ operator/(const XYZ& a, const XYZ& b)
{
	return XYZ(a.m_xyz[0] / b.m_xyz[0], a.m_xyz[1] / b.m_xyz[1], a.m_xyz[2] / b.m_xyz[2]);
}

inline XYZ operator/(const XYZ& a, const double b) {
	return XYZ(a.m_xyz[0] / b, a.m_xyz[1] / b, a.m_xyz[2] / b);
}

double XYZ::norm() const {
	return std::sqrt(m_xyz[0] * m_xyz[0] + m_xyz[1] * m_xyz[1] + m_xyz[2] * m_xyz[2]);
}

double XYZ::normSquared() const {
	return m_xyz[0] * m_xyz[0] + m_xyz[1] * m_xyz[1] + m_xyz[2] * m_xyz[2];
}

void XYZ::toUnit() {
	double norm = this->norm();
	if (norm != 0.0) {
		m_xyz[0] /= norm;
		m_xyz[1] /= norm;
		m_xyz[2] /= norm;
	} else {
		m_xyz[0] = 0.0;
		m_xyz[1] = 0.0;
		m_xyz[2] = 0.0;
	}
}

inline XYZ XYZ::unit() const {
	double norm = this->norm();
	if (norm != 0.0)
		return XYZ(m_xyz[0] / norm, m_xyz[1] / norm, m_xyz[2] / norm);
	else return XYZ{ 0.0, 0.0, 0.0 };
}

inline double dot(const XYZ& u, const XYZ& v) {
	return u.m_xyz[0] * v.m_xyz[0] + u.m_xyz[1] * v.m_xyz[1] + u.m_xyz[2] * v.m_xyz[2];
}

inline XYZ cross(const XYZ& u, const XYZ& v) {
	return XYZ { u.m_xyz[1] * v.m_xyz[2] - u.m_xyz[2] * v.m_xyz[1],
				 u.m_xyz[2] * v.m_xyz[0] - u.m_xyz[0] * v.m_xyz[2],
				 u.m_xyz[0] * v.m_xyz[1] - u.m_xyz[1] * v.m_xyz[0]
	};
}

XYZ& XYZ::rotateX(const XYZ& center, const double theta) {
	double thetaRad = DEG_TO_RAD(theta);
	Vector3 centerToPoint{ *this -center};
	m_xyz[0] = center.x() + dot(centerToPoint, XYZ{1.0, 0.0, 0.0});
	m_xyz[1] = center.y() + dot(centerToPoint, XYZ{0.0, cos(thetaRad), -sin(thetaRad)});
	m_xyz[2] = center.z() + dot(centerToPoint, XYZ{0.0, sin(thetaRad), cos(thetaRad)});
	return *this;
}

XYZ& XYZ::rotateY(const XYZ& center, const double theta) {
	double thetaRad = DEG_TO_RAD(theta);
	Vector3 centerToPoint{ *this -center};
	m_xyz[0] = center.x() + dot(centerToPoint, XYZ{cos(thetaRad), 0.0, sin(thetaRad)});
	m_xyz[1] = center.y() + dot(centerToPoint, XYZ{0.0, 1.0, 0.0});
	m_xyz[2] = center.z() + dot(centerToPoint, XYZ{-sin(thetaRad), 0.0, cos(thetaRad)});
	return *this;
}

XYZ& XYZ::rotateZ(const XYZ& center, const double theta) {
	double thetaRad = DEG_TO_RAD(theta);
	Vector3 centerToPoint{ *this -center};
	m_xyz[0] = center.x() + dot(centerToPoint, XYZ{cos(thetaRad), -sin(thetaRad), 0.0});
	m_xyz[1] = center.y() + dot(centerToPoint, XYZ{sin(thetaRad), cos(thetaRad), 0.0});
	m_xyz[2] = center.z() + dot(centerToPoint, XYZ{0.0, 0.0, 1.0});
	return *this;
}

XYZ& XYZ::rotateXYZ(const XYZ& center, const double alpha, const double beta, const double gamma) {
	return this->rotateZ(center, gamma).rotateY(center, beta).rotateX(center, alpha);
}

XYZ& XYZ::rotateAAxis(const XYZ& axis, const double theta) {
	double thetaRad = DEG_TO_RAD(theta);
	Vector3 axisCopy = axis.unit();
	Vector3 rotatedPoint = (*this) * cos(thetaRad) +
			(cross(axisCopy, *this) * sin(thetaRad)) +
			(axisCopy * dot(axisCopy, *this)*(1 - cos(thetaRad)));
	*this = rotatedPoint;
	return *this;
}

std::ostream& operator<<(std::ostream& out, const XYZ& xyz) {
	return out << "(x:" << xyz.m_xyz[0] << ", y:" << xyz.m_xyz[1] << ", z:" << xyz.m_xyz[2] << ")";
}
















