
#include <iostream>

#include "XYZ.h"

using namespace std;

int main() {

	Vector3 vi { 1.0, 0.0, 0.0 };
	Vector3 vj { 0.0, 1.0, 0.0 };
	Vector3 vk { 0.0, 0.0, 1.0 };

	Vector3 r1 { 0.1, 2.3, 1.0 };
	Vector3 r2 { -2.6, -2.9, 3.0 };
	Vector3 r3 { 1.8, -2.0, -1.0 };

	cout << "operator<<" << endl;
	cout << vi << vj << vk << endl;
	cout << endl;

	cout << "operator[]" << endl;
	for (int i=0; i<3; i++)
		cout << vi[i] << endl;
	cout << endl;

	cout << "operator-" << endl;
	cout << -vi << -vj << -vk << endl;
	cout << endl;


	cout << "norm()" << endl;
	cout << "vi.norm()=" << vi.norm() << " vj.norm()=" << vj.norm() << " vk.norm()=" << vk.norm() << endl;
	cout << "r1.norm()=" << r1.norm() << " r2.norm()=" << r2.norm() << " r3.norm()=" << r3.norm() << endl;
	cout << endl;

	cout << "normSquared()" << endl;
	cout << "vi.normSquared()=" << vi.normSquared() << " vj.normSquared()=" << vj.normSquared() << " vk.normSquared()=" << vk.normSquared() << endl;
	cout << "r1.normSquared()=" << r1.normSquared() << " r2.normSquared()=" << r2.normSquared() << " r3.normSquared()=" << r3.normSquared() << endl;
	cout << endl;







	return EXIT_SUCCESS;
}


