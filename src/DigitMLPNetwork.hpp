/*
 * Handwritten digit classifier
 * 
 * A handwritten digit classifier implemented 
 * with multi-layer perceptron
 *
 * Author: Marcos Lerones
 */

#pragma once

#include <fstream>

#include "MLP.hpp"

namespace artnn
{
    class DigitMLPNet
    {
    public:
        DigitMLPNet(uint hiddenNodes, double tEtha, double tAlpha)
        {
            std::vector<uint> sizes = { 784, hiddenNodes, 10 };
            network = new MLP<double>(sizes, tEtha, tAlpha);
            network->randomizeWeights(randomInitializer);
        }

        // Classify the given image
        // Returns -1 if it didn't recognize any digit
        int evaluate(const std::vector<double>& X);

        // Same as evaluate but different criteria
        int evaluateAux(const std::vector<double>& X);

        // Train for a specified number of epochs
        // or until error is less than some epsilon
        // Return number of epochs trained
        int trainFull(const std::vector<std::pair<std::vector<double>, int>>& trainingData,
                      const std::vector<std::pair<std::vector<double>, int>>& validationData,
                      const uint maxEpochs = 50, const double epsilon = 0.0,
                      const std::string trainLogFileName = "trainError.csv",
                      const std::string validLogFileName = "validError.csv");

        // Count the number of correctly classified digits
        int test(const std::vector<std::pair<std::vector<double>, int>>& testingData);

        // Count the number of correctly classified digits
        int testAux(const std::vector<std::pair<std::vector<double>, int>>& testingData);

    private:
        MLP<double>* network;

        // Transform digit into canonical vector of size 10
        // with 1 on the digit index and 0 everywhere else
        std::vector<double> digitToVector(int d);

        // Helper function to init weights
        static double randomInitializer();  
    };
};