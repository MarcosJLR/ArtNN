#include "ActivationFunctions.hpp"

namespace artnn
{
    template<typename T>
    T sgn(T x)
    {
        return x < 0 ? 0 : 1;  
    }

    template <typename T>
    T id(T x)
    {
        return x;
    }
};