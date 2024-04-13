
xyz-test.o: xyz-test.cpp
	gcc -c xyz-test.cpp -o xyz-test.o

XYZ.o: XYZ.cpp
	gcc -c XYZ.cpp -o XYZ.o
	
xyz-test: xyz-test.o XYZ.o
	g++ xyz-test.o XYZ.o -o xyz-test

all: xyz-test

test: all
	./xyz-test

