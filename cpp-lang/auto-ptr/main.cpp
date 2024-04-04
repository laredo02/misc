
#include <iostream>
#include <string>

using std::cout;
using std::endl;
using std::string;

template<typename T>class auto_ptr {
public:
	auto_ptr(T* ptr=nullptr): m_ptr(ptr) {}
	~auto_ptr() { delete m_ptr; }
	
	T& operator*() const { return *m_ptr; }
	T* operator->() const { return m_ptr; }
private:
	T* m_ptr;
};

int main () {

	auto_ptr<string> autoptr( new string{"autoptr"} );
	string* ptr { new string{"classicptr"} };

	cout << *autoptr << endl;
	cout << *ptr << endl;

	delete ptr;
	ptr = nullptr;

	return 0;
}


