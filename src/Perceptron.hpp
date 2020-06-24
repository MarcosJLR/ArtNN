/*
 * Perceptron neuron
 *
 * A simple perceptron implementation using
 * a templated general neuron
 *
 * Author: Marcos Lerones
 */

#pragma once

#include "Neuron.hpp"
#include "ActivationFunctions.hpp"

namespace artnn
{
    template <typename T>
    class Perceptron : public Neuron<T>
    {
    public:
        // Constructor
        Perceptron(uint tSize, T tEtha)
            : Neuron<T>(tSize, sgn<T>, tEtha) {}

        // Perceptron training function
        bool train(const std::vector<T>& X, const T desiredOutput) override;
    };
};