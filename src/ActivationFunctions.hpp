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

#include <cmath>

namespace artnn
{
    // Sign function: 0 if x < 0 and 1 otherwise
    template <typename T>
    inline T sgn(T x)
    {
        return x < 0 ? 0 : 1;  
    }

    // Identity function
    template <typename T>
    inline T id(T x)
    {
    	return x;
    } 

    // Logistic function (a = num / den)
    template <typename T, int num = 1, int den = 1>
    inline T logistic(T x)
    {
        T a = static_cast<T>(num) / static_cast<T>(den);
        return 1 / (1 + exp(-a*x));
    } 
};