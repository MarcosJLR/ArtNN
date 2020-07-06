#include "MLP.hpp" 

#include <iostream>

namespace artnn
{
    template <typename T>
    std::vector<T> MLPNeuron<T>::train(const std::vector<T>& X, const T errorSignal, const T y)
    {
        // Return vector (we ommit the bias)
        std::vector<T> backResult(Neuron<T>::mSize - 1);

        T activationDerivate = mLogisticConst * y * (1 - y);
        T localGradient = errorSignal * activationDerivate;

        for(uint i = 0; i < backResult.size(); i++)
        {
            backResult[i] = Neuron<T>::mWeights[i] * localGradient;
        }

        for(uint i = 0; i < Neuron<T>::mSize; i++)
        {
            mLastDeltaW[i] = mAlpha * mLastDeltaW[i] + Neuron<T>::mEtha * localGradient * X[i];
            Neuron<T>::mWeights[i] += mLastDeltaW[i];
        }

        return backResult;
    }

    template <typename T>
    std::vector<T> MLPLayer<T>::evaluate(const std::vector<T>& X)
    {
        std::vector<T> output(mOutputSize);
        for(uint i = 0; i < mOutputSize; i++)
        {
            output[i] = mNeuron[i].evaluate(X);
        }

        return output;
    }

    template <typename T>
    std::vector<T> MLPLayer<T>::train(const std::vector<T>& X, const std::vector<T>& errorSignal,
                                      const std::vector<T>& Y)
    {
        std::vector<T> backResult(mInputSize - 1, 0);

        for(uint i = 0; i < mOutputSize; i++)
        {
            std::vector<T> localErrorSignal = mNeuron[i].train(X, errorSignal[i], Y[i]);
            for(uint j = 0; j < backResult.size(); j++)
            {
                backResult[j] += localErrorSignal[j];
            }
        }

        return backResult;
    }

    template <typename T>
    std::vector<T> MLP<T>::evaluate(const std::vector<T>& X)
    {
        std::vector<T> Y = X;   

        for(uint i = 0; i < mLayer.size(); i++)
        {
            Y.push_back(1);
            Y = mLayer[i]->evaluate(Y);
        }

        return Y;
    }

    template <typename T>
    T MLP<T>::train(const std::vector<T>& X, const std::vector<T>& D)
    {
        std::vector<std::vector<T>> Y(mSize);
        std::vector<T> e(D.size());
        T sqError = 0;
        Y[0] = X;

        for(uint i = 0; i < mLayer.size(); i++)
        {
            Y[i].push_back(1);
            Y[i+1] = mLayer[i]->evaluate(Y[i]);
        }

        for(uint i = 0; i < D.size(); i++)
        {
            e[i] = D[i] - Y[mSize-1][i];
            sqError += e[i] * e[i];
        }

        for(uint i = mLayer.size(); i >= 1; i--)
        {
            std::vector<T> err;
            err = mLayer[i-1]->train(Y[i-1], e, Y[i]);
            e = err;
        }

        return 0.5 * sqError;
    }

    template <typename T>
    T MLP<T>::validate(const std::vector<T>& X, const std::vector<T>& D)
    {
        T sqError = 0;

        std::vector<T> Y = evaluate(X);

        for(uint i = 0; i < D.size(); i++)
        {
            T e = D[i] - Y[i];
            sqError += e * e;
        }

        return 0.5 * sqError;
    }
    

    template <typename T>
    void MLP<T>::randomizeWeights(std::function<T()> randFunc)
    {
        for(auto& L : mLayer)
        {
            L->randomizeWeights(randFunc);  
        }
    }

    template <typename T>
    void MLPLayer<T>::randomizeWeights(std::function<T()> randFunc)
    {
        for(auto& N : mNeuron)
        {
            N.randomizeWeights(randFunc);
        }
    }

    template <typename T>
    T MLP<T>::trainEpoch(std::vector<std::pair<std::vector<T>, std::vector<T>>>& trainingData)
    {
        random_shuffle(trainingData.begin(), trainingData.end());
        T avgError = 0.0;

        for(auto& [X, D] : trainingData)
        {
            avgError += train(X, D);
        }

        return avgError / (double) trainingData.size();
    }

    template <typename T>
    T MLP<T>::validateEpoch(std::vector<std::pair<std::vector<T>, std::vector<T>>>& validationData)
    {
        T avgError = 0.0;

        for(auto& [X, D] : validationData)
        {
            avgError += validate(X, D);
        }

        return avgError / (double) validationData.size();
    }

    template class MLPNeuron<float>;
    template class MLPNeuron<double>;
    template class MLPNeuron<long double>;

    template class MLPLayer<float>;
    template class MLPLayer<double>;
    template class MLPLayer<long double>;
    
    template class MLP<float>;
    template class MLP<double>;
    template class MLP<long double>;
};