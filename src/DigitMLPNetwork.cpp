#include "DigitMLPNetwork.hpp"

#include <iostream>

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

    int DigitMLPNet::evaluateAux(const std::vector<double>& X)
    {
        std::vector<double> Y = network->evaluate(X);

        int answer = -1;
        for(int i = 0; i < 10; i++)
        {
            if(Y[i] > 0.5)
            {
                if(answer == -1) { answer = i; }
                else { answer = -1; break; }
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
                               const std::vector<std::pair<std::vector<double>, int>>& rawValidData,
                               const uint maxEpochs, const double epsilon,
                               const std::string trainLogFileName,
                               const std::string validLogFileName)
    {
        std::vector<std::pair<std::vector<double>, std::vector<double>>> trainingData(rawData.size());
        std::vector<std::pair<std::vector<double>, std::vector<double>>> validationData(rawValidData.size());
        std::ofstream trainLog(trainLogFileName);
        std::ofstream validLog(validLogFileName);
        int totEpochs = maxEpochs;

        for(uint i = 0; i < rawData.size(); i++)
        {
            auto& [X, d] = rawData[i];
            trainingData[i] = { X, digitToVector(d) };
        }
        for(uint i = 0; i < rawValidData.size(); i++)
        {
            auto& [X, d] = rawValidData[i];
            validationData[i] = { X, digitToVector(d) };
        }

        for(uint i = 0; i < maxEpochs; i++)
        {
            double avgTrainError = network->trainEpoch(trainingData);
            trainLog << i << "," << avgTrainError << std::endl;

            double avgValidationError = network->validateEpoch(validationData);
            validLog << i << "," << avgValidationError << std::endl;

            if(avgTrainError < epsilon) { totEpochs = i; break; }
        }

        trainLog.close();
        validLog.close();

        return totEpochs;
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

    int DigitMLPNet::testAux(const std::vector<std::pair<std::vector<double>, int>>& testingData)
    {
        int answer = 0;
        for(auto& [X, d] : testingData)
        {
            int y = evaluateAux(X);
            if(y == d) { answer++; }
        }

        return answer;
    }

    double DigitMLPNet::randomInitializer()
    {
        static std::random_device rd;
        static std::mt19937 e2(rd());
        static std::uniform_real_distribution<double> dist(-0.05, 0.05);

        return dist(e2);
    }
};