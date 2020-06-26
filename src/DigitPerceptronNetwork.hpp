/*
 * Handwritten digit classifier
 * 
 * A Neural Network of Perceptron neurons made to
 * classify handwritten digits represented in an 
 * 28x28 pixels image of grayscale codes
 *
 * Author: Marcos Lerones
 */

#pragma once

#include <random>

#include "Perceptron.hpp"

namespace artnn
{
    class DigitPerceptronNet
    {
    public:
        // Constructor
        DigitPerceptronNet(double tEtha)
        {
            for(uint i = 0; i < 10; i++)
            {
                mNeurons[i] = new Perceptron<double>(785, tEtha);
                mNeurons[i]->randomizeWeights(randomInitializer);
            }
        }

        // Train this network with a data point
        // Returns mean squared error of all neurons
        double train(const std::vector<double>& X, int answer); 

        // Train this network with data set for an epoch
        // Returns mean squared error of all data
        double trainEpoch(std::vector<std::pair<std::vector<double>, int>>& input);

        // Train for a given number of epochs or until
        // mean squared error is less than epsilon
        // Returns number of epochs trained
        int trainFull(std::vector<std::pair<std::vector<double>, int>>& input,
                      const uint maxEpoch = 50, const double eps = 0.0);

        // Feed input to Network and return classification
        // Returns -1 if it doesn't recognize any digit
        int evaluate(std::vector<double>& X);

        // Test a batch of data and return the number of 
        // samples classified correctly
        int test(std::vector<std::pair<std::vector<double>, int>>& input);

    private:
        Perceptron<double>* mNeurons[10];      // Output Neurons

        static double randomInitializer();  // Helper function to init weights
    };
};