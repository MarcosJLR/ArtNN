#include <iostream>

#include "CSVReader.hpp"
#include "DigitAdalineNetwork.hpp"
#include "DigitPerceptronNetwork.hpp"

void storeData(std::vector<std::pair<std::vector<double>, int>>& finalData,
               std::vector<std::vector<std::string>> rawData)
{
    finalData.clear();

    for(auto &line : rawData)
    {
        std::vector<double> v;
        int digit = -1;
        for(auto& s : line)
        {
            if(digit == -1)
            {
                digit = std::stoi(s);
                v.push_back(1.0);       // Bias extra spot
            }
            else
            {
                v.push_back(std::stod(s) / 255.0);
            }
        }

        finalData.push_back({v, digit});
    }
}

int main(int argc, char* argv[])
{
    if(argc < 3)
    {
        std::cout << "Usage: ./DigitTester <training_data_csv> <testing_data_csv>\n";
        return 0;
    }       

    std::string trainingFilename = argv[1];
    std::string testingFilename = argv[2];

    artnn::CSVReader trainingReader(trainingFilename);
    artnn::CSVReader testingReader(testingFilename);

    std::vector<std::pair<std::vector<double>, int>> trainingData;
    std::vector<std::pair<std::vector<double>, int>> testingData;

    std::cout << "Reading Training Data...\n";
    storeData(trainingData, trainingReader.getData());

    std::cout << "Reading Testing Data...\n";
    storeData(testingData, testingReader.getData());

    artnn::DigitPerceptronNet perceptron0(0.001);
    artnn::DigitPerceptronNet perceptron1(0.01);
    artnn::DigitPerceptronNet perceptron2(0.1);

    artnn::DigitAdalineNet adaline0(0.0001);
    artnn::DigitAdalineNet adaline1(0.001);
    artnn::DigitAdalineNet adaline2(0.01);

    std::cout << "\nTraining Perceptron Network 0 for 50 epochs (etha = 0.001)...\n";
    perceptron0.trainFull(trainingData, 50, 0);
    std::cout << "Training finished succesfully\n";

    std::cout << "\nTraining Perceptron Network 1 for 50 epochs (etha = 0.01)...\n";
    perceptron1.trainFull(trainingData, 50, 0);
    std::cout << "Training finished succesfully\n";

    std::cout << "\nTraining Perceptron Network 2 for 50 epochs (etha = 0.1)...\n";
    perceptron2.trainFull(trainingData, 50, 0);
    std::cout << "Training finished succesfully\n";

    std::cout << "\nTraining Adaline Network 0 for 50 epochs (etha = 0.0001)...\n";
    adaline0.trainFull(trainingData, 50, 0);
    std::cout << "Training finished succesfully\n";

    std::cout << "\nTraining Adaline Network 1 for 50 epochs (etha = 0.001)...\n";
    adaline1.trainFull(trainingData, 50, 0);
    std::cout << "Training finished succesfully\n";

    std::cout << "\nTraining Adaline Network 2 for 50 epochs (etha = 0.01)...\n";
    adaline2.trainFull(trainingData, 50, 0);
    std::cout << "Training finished succesfully\n";

    int hitCount = -1;

    std::cout << "\nTesting Perceptron Network 0 (etha = 0.001)...\n";
    hitCount = perceptron0.test(testingData);
    std::cout << "Classified correctly " << hitCount << " out of " << testingData.size() << "\n";

    std::cout << "\nTesting Perceptron Network 1 (etha = 0.01)...\n";
    hitCount = perceptron1.test(testingData);
    std::cout << "Classified correctly " << hitCount << " out of " << testingData.size() << "\n";

    std::cout << "\nTesting Perceptron Network 2 (etha = 0.1)...\n";
    hitCount = perceptron2.test(testingData);
    std::cout << "Classified correctly " << hitCount << " out of " << testingData.size() << "\n";

    std::cout << "\nTesting Adaline Network 0 (etha = 0.0001)...\n";
    hitCount = adaline0.test(testingData);
    std::cout << "Classified correctly " << hitCount << " out of " << testingData.size() << "\n";

    std::cout << "\nTesting Adaline Network 1 (etha = 0.001)...\n";
    hitCount = adaline1.test(testingData);
    std::cout << "Classified correctly " << hitCount << " out of " << testingData.size() << "\n";

    std::cout << "\nTesting Adaline Network 2 (etha = 0.01)...\n";
    hitCount = adaline2.test(testingData);
    std::cout << "Classified correctly " << hitCount << " out of " << testingData.size() << "\n";
}
