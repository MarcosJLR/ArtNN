/*
 * Adaline Neuron
 *
 * Simple implementation of an Adaline Neuron
 * using templated neuron super class
 *
 * Author: Marcos Lerones
 */

#pragma once

#include "Neuron.hpp"
#include "ActivationFunctions.hpp"

namespace artnn
{
    template <typename T>
    class Adaline : public Neuron<T>
    {
    public:
        // Constructor
        Adaline(uint tSize, T tEtha)
            : Neuron<T>(tSize, id<T>, tEtha) {}

        // Adaline training function
        bool train(const std::vector<T>& X, const T desiredOutput);
    };
};