/*
 * Function Interpolator
 *
 * A Function Polynomial Interpolator usign Adaline
 * Neural Network.
 *
 */

#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <string>

#include "Adaline.hpp"

namespace artnn
{
    template <typename T>
    class Interpolator : Adaline<T>
    {
    public:
        // Constructor
        Interpolator(uint degree, T etha) 
            : Adaline<T>(degree + 1, etha) {}

        // Train a single epoch with a complete data set
        // Return mean squared error
        T trainEpoch(std::vector<std::pair<std::vector<T>,T>>& input);

        // Train for given number of epochs or until error
        // is less than epsilon
        // Return mean squared error after last epoch
        T trainFull(std::vector<std::pair<T, T>>& input, 
                      uint maxEpoch = 50, T epsilon = 0.0);

        // Evaluates polynomial in x
        inline T eval(T x) { return this->evaluate(createPolyVector(x)); } 

        // Plot into a CSV file
        void plotPoints(std::vector<T>& X, std::string filename);

    private:
        // Helper functions
        // Create vector { 1, x^1, x^2, x^3, ... }
        std::vector<T> createPolyVector(T x);
    };
};