.PHONY: all clean

CC = g++
CFLAGS = -std=c++17 -O2 -Wall
OBJS = Neuron.o CSVReader.o Perceptron.o Adaline.o DigitAdalineNetwork.o DigitPerceptronNetwork.o

all: DigitTester clean

DigitTester: src/DigitTester.cpp $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS)

Neuron.o: src/Neuron.cpp src/Neuron.hpp 
	$(CC) $(CFLAGS) -c -o $@ $< 

CSVReader.o: src/CSVReader.cpp src/CSVReader.hpp
	$(CC) $(CFLAGS) -c -o $@ $<

Perceptron.o: src/Perceptron.cpp src/Perceptron.hpp src/ActivationFunctions.hpp
	$(CC) $(CFLAGS) -c -o $@ $<

Adaline.o: src/Adaline.cpp src/Adaline.hpp src/ActivationFunctions.hpp
	$(CC) $(CFLAGS) -c -o $@ $<

DigitPerceptronNetwork.o: src/DigitPerceptronNetwork.cpp src/DigitPerceptronNetwork.hpp
	$(CC) $(CFLAGS) -c -o $@ $<

DigitAdalineNetwork.o: src/DigitAdalineNetwork.cpp src/DigitAdalineNetwork.hpp
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm *.o