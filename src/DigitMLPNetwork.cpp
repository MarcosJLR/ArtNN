#include "DigitMLPNetwork.hpp"

namespace artnn
{
    int DigitMLPNet::evaluate(const std::vector<double>& X)
    {
        std::vector<double> Y = network->evaluate(X);

        int answer = -1;
        for(int i = 0; i < 10; i++)
        {
            if(Y[i] > 0.9)
            {
                if(answer == -1) { answer = i; }
                else { answer = -1; break; }
            }
            else if(Y[i] > 0.1)
            {
                answer = -1;
                break;
            }
        }

        return answer;
    }

    std::vector<double> DigitMLPNet::digitToVector(int d)
    {
        std::vector<double> v(10, 0.0);
        v[d] = 1.0;

        return v;
    }

    int DigitMLPNet::trainFull(const std::vector<std::pair<std::vector<double>, int>>& rawData,
                               const uint maxEpochs, const double epsilon,
                               const std::string logFileName)
    {
        std::vector<std::pair<std::vector<double>, std::vector<double>>> trainingData(rawData.size());
        std::ofstream logFile(logFileName);

        for(uint i = 0; i < rawData.size(); i++)
        {
            auto& [X, d] = rawData[i];
            trainingData[i] = { X, digitToVector(d) };
        }

        for(uint i = 0; i < maxEpochs; i++)
        {
            double avgError = network->trainEpoch(trainingData);
            logFile << i << "," << avgError << std::endl;

            if(avgError < epsilon) { return i; }
        }

        logFile.close();

        return maxEpochs;
    }

    int DigitMLPNet::test(const std::vector<std::pair<std::vector<double>, int>>& testingData)
    {
        int answer = 0;
        for(auto& [X, d] : testingData)
        {
            int y = evaluate(X);
            if(y == d) { answer++; }
        }

        return answer;
    }
};