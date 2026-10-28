CXX ?= c++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra -Wpedantic
all: build/kfnnue
build/kfnnue: src/main.cpp src/stream.h
	mkdir -p build
	$(CXX) $(CXXFLAGS) $< -o $@
test: build/kfnnue
	./build/kfnnue selftest
	python3 -m unittest discover -s tests -v
clean:
	rm -rf build
.PHONY: all test clean

