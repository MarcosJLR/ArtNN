#include "Perceptron.hpp"

namespace artnn
{
    template <typename T>
    bool Perceptron<T>::train(const std::vector<T>& X, T desiredOutput)
    {
        T y = Neuron<T>::evaluate(X);
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

    template class Perceptron<float>;
    template class Perceptron<double>;
    template class Perceptron<long double>;
    template class Perceptron<int>;
    template class Perceptron<long long>;
};