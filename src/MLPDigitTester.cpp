#include <iostream>
#include <fstream>

#include "CSVReader.hpp"
#include "DigitMLPNetwork.hpp"

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

    srand(time(0));

    int hitCount = -1;

    std::string trainingFilename = argv[1];
    std::string testingFilename = argv[2];

    artnn::CSVReader trainingReader(trainingFilename);
    artnn::CSVReader testingReader(testingFilename);

    std::vector<std::pair<std::vector<double>, int>> trainingData;
    std::vector<std::pair<std::vector<double>, int>> testingData;

    std::vector<std::pair<std::vector<double>, int>> trainingData6;
    std::vector<std::pair<std::vector<double>, int>> trainingData7;

    std::cout << "Reading Training Data...\n";
    storeData(trainingData, trainingReader.getData());

    std::cout << "Reading Testing Data...\n";
    storeData(testingData, testingReader.getData());

    random_shuffle(trainingData.begin(), trainingData.end());
    for(uint i = 0; i < trainingData.size() / 4; i++)
    {
        trainingData6.push_back(trainingData[i]);
    }

    random_shuffle(trainingData.begin(), trainingData.end());
    for(uint i = 0; i < trainingData.size() / 2; i++)
    {
        trainingData7.push_back(trainingData[i]);
    }

    artnn::DigitMLPNet net0(20, 0.1, 0.9);
    artnn::DigitMLPNet net1(50, 0.1, 0.9);
    artnn::DigitMLPNet net2(100, 0.1, 0.9);

    artnn::DigitMLPNet net3(100, 0.1, 0.0);
    artnn::DigitMLPNet net4(100, 0.1, 0.25);
    artnn::DigitMLPNet net5(100, 0.1, 0.5);

    artnn::DigitMLPNet net6(100, 0.1, 0.9);
    artnn::DigitMLPNet net7(100, 0.1, 0.9);

    // -------------------- Network 0 ---------------------- //

    std::cout << "\nTraining Network 0 for 50 epochs (n = 20, etha = 0.1, alpha = 0.9)...\n";
    net0.trainFull(trainingData, testingData, 50, 0, "Logs/MLP/1/trainError0.csv", "Logs/MLP/1/validError0.csv");
    std::cout << "Training finished succesfully\n";

    std::cout << "\nTesting Network 0 (n = 20, etha = 0.1, alpha = 0.9)...\n";
    hitCount = net0.test(testingData);
    std::cout << "Classified correctly " << hitCount << " out of " << testingData.size() << "\n";

    std::cout << "\nTesting Network 0 second criteria (n = 20, etha = 0.1, alpha = 0.9)...\n";
    hitCount = net0.testAux(testingData);
    std::cout << "Classified correctly " << hitCount << " out of " << testingData.size() << "\n";

    // -------------------- Network 0 ---------------------- //

    // -------------------- Network 1 ---------------------- //

    std::cout << "\nTraining Network 1 for 50 epochs (n = 50, etha = 0.1, alpha = 0.9)...\n";
    net1.trainFull(trainingData, testingData, 50, 0, "Logs/MLP/1/trainError1.csv", "Logs/MLP/1/validError1.csv");
    std::cout << "Training finished succesfully\n";

    std::cout << "\nTesting Network 1 (n = 50, etha = 0.1, alpha = 0.9)...\n";
    hitCount = net1.test(testingData);
    std::cout << "Classified correctly " << hitCount << " out of " << testingData.size() << "\n";

    std::cout << "\nTesting Network 1 second criteria (n = 50, etha = 0.1, alpha = 0.9)...\n";
    hitCount = net1.testAux(testingData);
    std::cout << "Classified correctly " << hitCount << " out of " << testingData.size() << "\n";

    // -------------------- Network 1 ---------------------- //

    // -------------------- Network 2 ---------------------- //

    std::cout << "\nTraining Network 2 for 50 epochs (n = 100, etha = 0.1, alpha = 0.9)...\n";
    net2.trainFull(trainingData, testingData, 50, 0, "Logs/MLP/1/trainError2.csv", "Logs/MLP/1/validError2.csv");
    std::cout << "Training finished succesfully\n";

    std::cout << "\nTesting Network 2 (n = 100, etha = 0.1, alpha = 0.9)...\n";
    hitCount = net2.test(testingData);
    std::cout << "Classified correctly " << hitCount << " out of " << testingData.size() << "\n";

    std::cout << "\nTesting Network 2 second criteria (n = 100, etha = 0.1, alpha = 0.9)...\n";
    hitCount = net2.testAux(testingData);
    std::cout << "Classified correctly " << hitCount << " out of " << testingData.size() << "\n";

    // -------------------- Network 2 ---------------------- //
    
    // -------------------- Network 3 ---------------------- //

    std::cout << "\nTraining Network 3 for 50 epochs (n = 100, etha = 0.1, alpha = 0.0)...\n";
    net3.trainFull(trainingData, testingData, 50, 0, "Logs/MLP/2/trainError3.csv", "Logs/MLP/2/validError3.csv");
    std::cout << "Training finished succesfully\n";

    std::cout << "\nTesting Network 3 (n = 100, etha = 0.1, alpha = 0.0)...\n";
    hitCount = net3.test(testingData);
    std::cout << "Classified correctly " << hitCount << " out of " << testingData.size() << "\n";

    std::cout << "\nTesting Network 3 second criteria (n = 100, etha = 0.1, alpha = 0.0)...\n";
    hitCount = net3.testAux(testingData);
    std::cout << "Classified correctly " << hitCount << " out of " << testingData.size() << "\n";

    // -------------------- Network 3 ---------------------- //

    // -------------------- Network 4 ---------------------- //

    std::cout << "\nTraining Network 4 for 50 epochs (n = 100, etha = 0.1, alpha = 0.25)...\n";
    net4.trainFull(trainingData, testingData, 50, 0, "Logs/MLP/2/trainError4.csv", "Logs/MLP/2/validError4.csv");
    std::cout << "Training finished succesfully\n";

    std::cout << "\nTesting Network 4 (n = 100, etha = 0.1, alpha = 0.25)...\n";
    hitCount = net4.test(testingData);
    std::cout << "Classified correctly " << hitCount << " out of " << testingData.size() << "\n";

    std::cout << "\nTesting Network 4 second criteria (n = 100, etha = 0.1, alpha = 0.25)...\n";
    hitCount = net4.testAux(testingData);
    std::cout << "Classified correctly " << hitCount << " out of " << testingData.size() << "\n";

    // -------------------- Network 4 ---------------------- //
     
    // -------------------- Network 5 ---------------------- //

    std::cout << "\nTraining Network 5 for 50 epochs (n = 100, etha = 0.1, alpha = 0.5)...\n";
    net5.trainFull(trainingData, testingData, 50, 0, "Logs/MLP/2/trainError5.csv", "Logs/MLP/2/validError5.csv");
    std::cout << "Training finished succesfully\n";

    std::cout << "\nTesting Network 5 (n = 100, etha = 0.1, alpha = 0.5)...\n";
    hitCount = net5.test(testingData);
    std::cout << "Classified correctly " << hitCount << " out of " << testingData.size() << "\n";

    std::cout << "\nTesting Network 5 second criteria (n = 100, etha = 0.1, alpha = 0.5)...\n";
    hitCount = net5.testAux(testingData);
    std::cout << "Classified correctly " << hitCount << " out of " << testingData.size() << "\n";

    // -------------------- Network 5 ---------------------- //

    // -------------------- Network 6 ---------------------- //

    std::cout << "\nTraining Network 6 for 50 epochs and 1/4 of the data (n = 100, etha = 0.1, alpha = 0.9)...\n";
    net6.trainFull(trainingData6, testingData, 50, 0, "Logs/MLP/3/trainError6.csv", "Logs/MLP/3/validError6.csv");
    std::cout << "Training finished succesfully\n";

    std::cout << "\nTesting Network 6 (n = 100, etha = 0.1, alpha = 0.9)...\n";
    hitCount = net6.test(testingData);
    std::cout << "Classified correctly " << hitCount << " out of " << testingData.size() << "\n";

    std::cout << "\nTesting Network 6 second criteria (n = 100, etha = 0.1, alpha = 0.9)...\n";
    hitCount = net6.testAux(testingData);
    std::cout << "Classified correctly " << hitCount << " out of " << testingData.size() << "\n";

    // -------------------- Network 6 ---------------------- //
     
    // -------------------- Network 7 ---------------------- //

    std::cout << "\nTraining Network 7 for 50 epochs and 1/2 of the data (n = 100, etha = 0.1, alpha = 0.9)...\n";
    net7.trainFull(trainingData7, testingData, 50, 0, "Logs/MLP/3/trainError7.csv", "Logs/MLP/3/validError7.csv");
    std::cout << "Training finished succesfully\n";

    std::cout << "\nTesting Network 7 (n = 100, etha = 0.1, alpha = 0.9)...\n";
    hitCount = net7.test(testingData);
    std::cout << "Classified correctly " << hitCount << " out of " << testingData.size() << "\n";

    std::cout << "\nTesting Network 7 second criteria (n = 100, etha = 0.1, alpha = 0.9)...\n";
    hitCount = net7.testAux(testingData);
    std::cout << "Classified correctly " << hitCount << " out of " << testingData.size() << "\n";

    // -------------------- Network 7 ---------------------- //

	return 0;	
}