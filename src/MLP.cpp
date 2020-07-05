#include "MLP.hpp" 

namespace artnn
{
	template <typename T>
	std::vector<T> MLPNeuron<T>::train(const std::vector<T>& X, const T errorSignal)
	{
		// Return vector (we ommit the bias)
		std::vector<T> backResult(Neuron<T>::mSize - 1);

		T y = Neuron<T>::evaluate(X);
		T activationDerivate = mLogisticConst * y * (1 - y);
		T localGradient = errorSignal * activationDerivate;

		for(uint i = 0; i + 1 < Neuron<T>::mSize; i++)
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
	std::vector<T> MLPLayer<T>::train(const std::vector<T>& X, const std::vector<T>& errorSignal)
	{
		std::vector<T> backResult(mInputSize - 1, 0);

		for(uint i = 0; i < mOutputSize; i++)
		{
			std::vector<T> localErrorSignal = mNeuron[i].train(X, errorSignal[i]);
			for(uint j = 0; j + 1 < mInputSize; j++)
			{
				backResult[j] += localErrorSignal[i];
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
		std::vector<std::vector<T>> Y(mLayer.size() + 1);
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
			e[i] = D[i] - Y[mSize][i];
			sqError += e[i] * e[i];
		}

		for(uint i = mSize; i >= 1; i--)
		{
			e = mLayer[i-1]->train(Y[i-1], e);
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
};