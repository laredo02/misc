
image-test.o: image-test.cpp Image.h
	gcc -c image-test.cpp -o image-test.o

image-test: image-test.o
	g++ image-test.o -o image-test

all: image-test

test: all
	./image-test



