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
		// Get output of feeding input Vector X to the neuron
		T evaluate(const std::vector<T>& X);

		// Randomize Synaptic Weights Vector
		void randomizeWeights(std::function<T()> randFunc);

		// Train function for one input with known desired output
		// To be implemented in subclasses 
		virtual bool train(const std::vector<T>& X, const T desiredOutput) = 0;

	protected:
		uint mSize;					// Size of the Weights Vector
		std::function<T(T)> mSigma; // Activation Function
		std::vector<T> mWeights; 	// Synaptic Weights
		T mEtha; 					// Learning Factor

		// Dot product between two vectors
		static T dotProduct(const std::vector<T>& X, const std::vector<T>& Y);
	};
};