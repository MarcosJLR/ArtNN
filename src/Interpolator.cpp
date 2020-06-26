#include "Interpolator.hpp"

namespace artnn
{
    template <typename T>
    T Interpolator<T>::trainEpoch(std::vector<std::pair<std::vector<T>,T>>& input)
    {
        // Randomize input
        std::random_shuffle(input.begin(), input.end());

        T acumError = 0;
        for(auto& [X, d] : input)
        {
            this->train(X, d);
            T y = this->evaluate(X);
            T sqError = (d - y) * (d - y);

            acumError += sqError;
        }

        return acumError / static_cast<T>(input.size());
    }

    template <typename T>
    T Interpolator<T>::trainFull(std::vector<std::pair<T,T>>& input,
                                 uint maxEpoch, T epsilon)
    {
        std::vector<std::pair<std::vector<T>, T>> processedInput;

        for(auto& [x, y] : input)
        {
            processedInput.push_back({createPolyVector(x), y});
        }

        T sqError = 0;
        for(uint i = 0; i < maxEpoch; i++)
        {
            sqError = trainEpoch(processedInput);
            if(sqError <= epsilon) 
                return sqError;
        }

        return sqError;
    }

    template <typename T>
    std::vector<T> Interpolator<T>::createPolyVector(T x)
    {
        std::vector<T> polyVector(this->mSize);
        polyVector[0] = static_cast<T>(1);

        for(uint i = 1; i < this->mSize; i++)
        {
            polyVector[i] = polyVector[i-1] * x;
        }

        return polyVector;
    }

    template <typename T>
    void Interpolator<T>::plotPoints(std::vector<T>& X, std::string filename)
    {
        std::ofstream file(filename);

        for(auto& x : X)
        {
            file << x << "," << eval(x) << "\n";
        }

        file.close();
    }

    template class Interpolator<float>;
    template class Interpolator<double>;
    template class Interpolator<long double>;
};