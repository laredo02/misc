
#include <iostream>
#include <string>

using std::cout;
using std::endl;

template<typename T> class List {
public:
	List(size_t size) : s{size}, ptr{new T[size]} {
		cout << this << "\tList::~List() -- Contructor" << endl;
	}

	~List() {
		cout << this << "\tList::~List() -- Destructor" << endl;
		delete[] ptr;
		ptr = nullptr;
	}

	List(const List& list) : s{list.s}, ptr{new T[list.s]} {
		cout << this << "\tList::List(const List&) -- Copy Constructor" << endl;
		for (size_t i{0}; i<s; i++)
			ptr[i] = list.ptr[i];
	}

	List(List&& list) noexcept : s{list.s}, ptr{list.ptr} {
		cout << this << "\tList::List(List&&) -- Move Constructor" << endl;
		list.s = 0;
		list.ptr = nullptr;
	}

	List& operator=(const List& list) {
		cout << this << "\tList::operator=(const List& list)" << endl;
		if (&list == this)
			return *this;

		delete[] ptr;
		ptr = nullptr;

		s = list.s;
		ptr = new T[s];

		for (size_t i{0}; i<s; i++)
			ptr[i] = list.ptr[i];

		return *this;
	}

	List& operator=(List&& list) {
		cout << this << "\tList::operator=(List&& list)" << endl;
		if(&list == nullptr)
			return *this;
			
		delete[] ptr;
		ptr = nullptr;

		s = list.s;
		ptr = list.ptr;
		list.s = 0;
		list.ptr = nullptr;

		return *this;

	}

	size_t size() const { return s; }
	T& operator[](size_t i) { return ptr[i]; }
	const T& operator[](size_t i) const { return ptr[i]; }

	template<typename U> friend std::ostream& operator<<(std::ostream& os, const List<U>& list);

private:
	T* ptr;
	size_t s;

};

template<typename U> std::ostream& operator<<(std::ostream& os, const List<U>& list) {
	std::string s {""};
	s += '{';
	for (int i=0; i<list.s; i++)
		s += list[i] + "\t";
	s += '}';
	return os << s;
}


int main() {

	List<int> list(10);
	List list2(list);
	List list3 = list2;
	List list4(std::move(list));

}









