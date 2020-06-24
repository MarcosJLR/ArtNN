#include "Adaline.hpp"

namespace artnn
{
    template <typename T>
    bool Adaline<T>::train(const std::vector<T>& X, T desiredOutput)
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

    template class Adaline<float>;
    template class Adaline<double>;
    template class Adaline<long double>;
    template class Adaline<int>;
    template class Adaline<long long>;
};