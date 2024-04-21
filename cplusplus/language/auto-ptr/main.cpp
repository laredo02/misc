
#include <iostream>
#include <string>
#include <memory>

using std::cout;
using std::endl;
using std::string;
using std::unique_ptr;

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

	auto_ptr(auto_ptr&& m) noexcept : m_ptr(m.m_ptr) {
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

	T& operator=(auto_ptr&& m) noexcept {
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

class A {
public:
	A() { cout << "A Constructor" << endl; }
	~A() { cout << "A Destructor" << endl; }
	friend std::ostream& operator<<(std::ostream& os, const A& a) { return os << "AAHAHAH..."; }
};

void uniqueCall(std::unique_ptr<A>& ptr) {
	if(ptr)
		cout << *ptr << endl;
}

int main () {

	auto_ptr<string> autoptr( new string{"autoptr"} );
	auto_ptr autocopy(autoptr);
	auto_ptr autocopyass = autoptr;

	cout << *autoptr << endl;
	cout << *autocopy << endl;
	cout << *autocopyass << endl;

	auto_ptr<string> autocopymove(std::move(autoptr));
	cout << *autocopymove << endl;
	
	unique_ptr<A> pa { std::make_unique<A>() };
	uniqueCall( pa );



	return 0;
}




