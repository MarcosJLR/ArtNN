#include "Neuron.hpp"

#ifdef WARNINGS_ENABLED 
#include <iostream>
#endif

namespace artnn
{
	template <typename T>
	T Neuron<T>::dotProduct(const std::vector<T>& X, const std::vector<T>& Y)
	{
		#ifdef WARNINGS_ENABLED
		if(X.size() != Y.size())
		{
			std::cerr << "Warning: Evaluating dot product of Vectors of different sizes\n";
		}
		#endif
		T answer = 0;
		for(uint i = 0; i < std::min(X.size(), Y.size()); i++)
		{
			answer += X[i] * Y[i];
		}
		return answer;
	}

	template <typename T>
	T Neuron<T>::evaluate(const std::vector<T>& X)
	{
		return mSigma(dotProduct(mWeights, X));	
	}

	template <typename T>
	void Neuron<T>::randomizeWeights(std::function<T()> randFunc)
	{
		for(T& w : mWeights)
		{
			w = randFunc();
		}
	}

	template class Neuron<float>;
	template class Neuron<double>;
	template class Neuron<long double>;
	template class Neuron<int>;
	template class Neuron<long long>;
};