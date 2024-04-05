
#include <iostream>
#include <string>

using std::cout;
using std::endl;
using std::string;

template<typename T>class auto_ptr {
public:
	auto_ptr(T* ptr=nullptr): m_ptr(ptr) {
		cout << "Constructor" << endl;
	}
	~auto_ptr() {
		cout << "Destructor" << endl;
		delete m_ptr;
	}

	auto_ptr(auto_ptr& copy): m_ptr(new T) {
		cout << "Copy Constructor" << endl;
		*m_ptr = *(copy.m_ptr);
	}

	auto_ptr(auto_ptr&& m): m_ptr(m.m_ptr) {
		cout << "Move Constructor" << endl;
		m.m_ptr = nullptr;
	}

	T& operator=(auto_ptr& copy) {
		cout << "Copy Assignment" << endl;
		if (&copy == this)
			return *this;
		delete m_ptr;
		*m_ptr = *(copy.m_ptr);
		return *this;
	}

	T& operator=(auto_ptr&& m)  {
		cout << "Move Assignment" << endl;
		if (&m == this)
			return *this;
		m_ptr = m.m_ptr;
		m.m_ptr = nullptr;
		return *this;
	}

	T& operator*() const { return *m_ptr; }
	T* operator->() const { return m_ptr; }

private:
	T* m_ptr;

};

int main () {

	auto_ptr<string> autoptr( new string{"autoptr"} );

	auto_ptr autocopy(autoptr);
	auto_ptr autocopyass = autoptr;

	std::cout << *autocopy << std::endl;
	std::cout << *autocopyass << std::endl;

	auto_ptr<string> autocopymove(std::move(autoptr));

	std::cout << *autocopymove << std::endl;
	

	return 0;
}




