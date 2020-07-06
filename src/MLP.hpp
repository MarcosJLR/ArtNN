/* 
 * A Multi-Layer Perceptron implementation
 * 
 * Template class of Multi-Layer perceptron
 * implemented using the abstract class 
 * Neuron.
 *
 * Author: Marcos Lerones
 */

#pragma once

#include <random>

#include "Neuron.hpp"
#include "ActivationFunctions.hpp"

namespace artnn
{
    template <typename T>
    class MLPNeuron : public Neuron<T>
    {
    public:
        MLPNeuron(uint tSize, T tEtha, T tAlpha = 0)
            : Neuron<T>(tSize, logistic<T,1,1>, tEtha), 
              mAlpha(tAlpha), mLastDeltaW(tSize, 0),
              mLogisticConst(1) 
        {}

        // Train this neuron given input and error signals
        // Return Vector of local gradient times weights 
        std::vector<T> train(const std::vector<T>& X, const T errorSignal, const T y);

    private:
        T mAlpha;                       // Momentum factor
        std::vector<T> mLastDeltaW;     // Last change to weights
        T mLogisticConst;               // Constant of the logistic function
    };

    template <typename T>
    class MLPLayer
    {
    public:
        MLPLayer(uint tInputSize, uint tOutputSize, T tEtha, T tAlpha = 0)
            : mInputSize(tInputSize), mOutputSize(tOutputSize),
              mNeuron(tOutputSize, MLPNeuron<T>(tInputSize, tEtha, tAlpha))
        {}

        // Get the output of feeding X to this layer
        std::vector<T> evaluate(const std::vector<T>& X);

        // Train this layer given input and error signals
        // Return Vector of sumation of local gradients times weights 
        std::vector<T> train(const std::vector<T>& X, const std::vector<T>& errorSignal,
                             const std::vector<T>& Y);

        // Initialize with random weights given by random
        // generating function
        void randomizeWeights(std::function<T()> randFunc);

    private:
        uint mInputSize;                    // Size of input
        uint mOutputSize;                   // Number of neurons
        std::vector<MLPNeuron<T>> mNeuron;  // Neurons
    };

    template <typename T>
    class MLP
    {
    public:
        MLP(std::vector<uint>& tSizes, T tEtha, T tAlpha = 0)
            : mSize(tSizes.size()), mLayer(tSizes.size() - 1, nullptr)
        {
            for(uint i = 1; i < tSizes.size(); i++)
            {
                mLayer[i-1] = new MLPLayer<T>(tSizes[i-1] + 1, tSizes[i], tEtha, tAlpha);
            }
        }

        // Feed input to the network and get result
        std::vector<T> evaluate(const std::vector<T>& X);

        // Train with input vector X and desired output D
        // Returns mean square error
        T train(const std::vector<T>& X, const std::vector<T>& D);

        // Train this Network for a complete epoch 
        // Return average mean square error
        T trainEpoch(std::vector<std::pair<std::vector<T>, std::vector<T>>>& trainingData);

        // Return mean square error
        T validate(const std::vector<T>& X, const std::vector<T>& D);

        // Return average mean square error
        T validateEpoch(std::vector<std::pair<std::vector<T>, std::vector<T>>>& validationData);

        // Initialize with random weights given by random
        // generating function
        void randomizeWeights(std::function<T()> randFunc);

    private:
        uint mSize;                         // Number of layers
        std::vector<MLPLayer<T>*> mLayer;   // Layers
    };
};