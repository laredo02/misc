
#include <iostream>

using namespace std;

#define HELLO -1

int main () {

#if HELLO==1
	cout << "Hello" << endl;
#elif HELLO==0
	cout << "Meh" << endl;
#elif HELLO==-1
	cout << "Goodbye" << endl;
#endif

	return 0;
}

