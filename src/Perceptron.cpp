#include "Perceptron.hpp"

namespace artnn
{
    template <typename T>
    bool Perceptron<T>::train(const std::vector<T>& X, T desiredOutput)
    {
        T y = evaluate(X);
        if(y != desiredOutput)
        {
            for(uint i = 0; i < Neuron<T>::mSize; i++)
            {
                Neuron<T>::mWeights[i] += Neuron<T>::mEtha * (desiredOutput - y) * X[i];
            }

            return true;
        }
        return false;
    }
};