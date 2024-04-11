
#include <iostream>
#include <vector>

using namespace std;

class Animal {
protected:
	Animal() { cout << "Animal::Animal()"; }
public:
	virtual void print() { cout << "Animal" << endl; }
};

class Cat: public Animal {
public:
	Cat() { cout << "Cat::Cat()" << endl; }
	void print() { cout << "Cat" << endl; }
};

class Dog: public Animal {
public:
	Dog() { cout << "Dog::Dog()" << endl; }
	void print() { cout << "Dog" << endl; }
};


int main () {

	vector<Animal*> animalList;

	animalList.push_back(new Cat());
	animalList.push_back(new Dog());
	animalList.push_back(new Cat());
	animalList.push_back(new Cat());
	animalList.push_back(new Dog());
	animalList.push_back(new Dog());

	for (auto it=animalList.begin(); it!=animalList.end(); ++it)
		(*it)->print();

	for (auto it=animalList.begin(); it!=animalList.end(); ++it)
		delete *it;

	return 0;
}

