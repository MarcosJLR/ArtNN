.PHONY: all clean

CC = g++
CFLAGS = -std=c++17 -O2 -Wall
OBJS = Neuron.o CSVReader.o Perceptron.o Adaline.o MLP.o DigitAdalineNetwork.o DigitPerceptronNetwork.o Interpolator.o DigitMLPNetwork.o

all: MLPDigitTester DigitTester PolyTester clean

MLPDigitTester: src/MLPDigitTester.cpp $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS)

DigitTester: src/DigitTester.cpp $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS)

PolyTester: src/PolyTester.cpp $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS)

Neuron.o: src/Neuron.cpp src/Neuron.hpp 
	$(CC) $(CFLAGS) -c -o $@ $< 

CSVReader.o: src/CSVReader.cpp src/CSVReader.hpp
	$(CC) $(CFLAGS) -c -o $@ $<

Perceptron.o: src/Perceptron.cpp src/Perceptron.hpp src/ActivationFunctions.hpp
	$(CC) $(CFLAGS) -c -o $@ $<

Adaline.o: src/Adaline.cpp src/Adaline.hpp src/ActivationFunctions.hpp
	$(CC) $(CFLAGS) -c -o $@ $<

MLP.o: src/MLP.cpp src/MLP.hpp src/ActivationFunctions.hpp
	$(CC) $(CFLAGS) -c -o $@ $<
	
Interpolator.o: src/Interpolator.cpp src/Interpolator.hpp src/Adaline.hpp
	$(CC) $(CFLAGS) -c -o $@ $<

DigitPerceptronNetwork.o: src/DigitPerceptronNetwork.cpp src/DigitPerceptronNetwork.hpp
	$(CC) $(CFLAGS) -c -o $@ $<

DigitAdalineNetwork.o: src/DigitAdalineNetwork.cpp src/DigitAdalineNetwork.hpp
	$(CC) $(CFLAGS) -c -o $@ $<

DigitMLPNetwork.o: src/DigitMLPNetwork.cpp src/DigitMLPNetwork.hpp
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm *.o
