/*
 * Common Activation Functions for Neurons
 *
 * Implementation to some of the most common 
 * Activation Functions used in Artificial
 * Neural Networks
 *
 * Author: Marcos Lerones
 */

#pragma once

namespace artnn
{
    // Sign function: 0 if x < 0 and 1 otherwise
    template <typename T>
    T sgn(T x);

    // Identity function
    template <typename T>
    T id(T x); 
};