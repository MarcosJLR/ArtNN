#include "DigitPerceptronNetwork.hpp"

namespace artnn
{
    double DigitPerceptronNet::randomInitializer()
    {
        static std::random_device rd;
        static std::mt19937 e2(rd());
        static std::uniform_real_distribution<double> dist(-0.05, 0.05);

        return dist(e2);
    }

    double DigitPerceptronNet::train(const std::vector<double>& X, int d)
    {
        double acumError = 0.0;
        for(int i = 0; i < 10; i++)
        {
            mNeurons[i]->train(X, i == d ? 1.0 : 0.0);

            double y = mNeurons[i]->evaluate(X);
            double dd = i == d ? 1.0 : 0.0;
            double sqError = (dd - y) * (dd - y);

            acumError += sqError;
        }
        return acumError / 10.0;
    }

    double DigitPerceptronNet::trainEpoch(std::vector<std::pair<std::vector<double>, int>>& input)
    {
        // Randomize input
        std::random_shuffle(input.begin(), input.end());

        double acumError = 0.0; 
        for(auto& [X, d] : input)
        {
            acumError += train(X, d);
        }

        return acumError / (double) input.size();
    }

    int DigitPerceptronNet::trainFull(std::vector<std::pair<std::vector<double>, int>>& input,
                                   const uint maxEpoch, const double eps)
    {
        for(uint i = 0; i < maxEpoch; i++)
        {
            if(trainEpoch(input) <= eps) return i+1;
        }

        return maxEpoch;
    }

    int DigitPerceptronNet::evaluate(std::vector<double>& X)
    {
        int answer = -1;
        int count = 0;

        for(uint i = 0; i < 10; i++)
        {
            double y = mNeurons[i]->evaluate(X);
            if(y == 1.0)
            {
                count++;
                answer = i;
            }
        }

        return count == 1 ? answer : -1;
    }

    int DigitPerceptronNet::test(std::vector<std::pair<std::vector<double>, int>>& input)
    {
        int count = 0;
        for(auto& [X, d] : input)
        {
            if(evaluate(X) == d)
            {
                count++;
            }
        }

        return count;
    }
};