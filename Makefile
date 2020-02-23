CXX=icpc
CXXFLAGS=-g -march=native -fopenmp -std=c++17 -O3 -Wall

all: a0

clean:
	rm -rf a0
