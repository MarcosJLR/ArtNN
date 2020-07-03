/*
 * A General Neuron model implementation 
 *
 * Abstract template class of a Neuron that 
 * can be used as a super class for specific 
 * neurons like perceptrons
 *
 * Author: Marcos Lerones
 */

#pragma once

#include <vector>
#include <functional>
#include <algorithm>

namespace artnn
{
    template <typename T>
    class Neuron
    {
    public:
        // Constructor
        Neuron(uint tSize, std::function<T(T)> tSigma, T tEtha)
            : mSize(tSize), mSigma(tSigma), mWeights(mSize), mEtha(tEtha) {}

        // Get output of feeding input Vector X to the neuron
        T evaluate(const std::vector<T>& X);

        // Randomize Synaptic Weights Vector
        void randomizeWeights(std::function<T()> randFunc);

    protected:
        uint mSize;                 // Size of the Weights Vector
        std::function<T(T)> mSigma; // Activation Function
        std::vector<T> mWeights;    // Synaptic Weights
        T mEtha;                    // Learning Factor

        // Dot product between two vectors
        static T dotProduct(const std::vector<T>& X, const std::vector<T>& Y);
    };
};