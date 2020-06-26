#include <iostream>

#include "CSVReader.hpp"
#include "Interpolator.hpp"

void storeData(std::vector<std::pair<double, double>>& finalData,
               std::vector<std::vector<std::string>> rawData)
{
    finalData.clear();

    for(auto& line : rawData)
    {
        finalData.push_back({std::stod(line[0]), std::stod(line[1])});
    }
}

int main(int argc, char* argv[])
{
    if(argc != 2)
    {
        std::cout << "Usage: ./PolyTester <data_csv>\n";
        return 0;
    }   

    std::string dataFilename = argv[1];

    artnn::CSVReader dataReader(dataFilename);

    std::vector<std::pair<double, double>> data;
    std::vector<double> dataX;

    std::cout << "Reading data..\n";
    storeData(data, dataReader.getData());

    for(auto& [x, y] : data)
    {
        dataX.push_back(x);
    }

    artnn::Interpolator<double> interpol0(3, 0.001);
    artnn::Interpolator<double> interpol1(4, 0.001);
    artnn::Interpolator<double> interpol2(5, 0.001);

    artnn::Interpolator<double> interpol3(3, 0.0001);
    artnn::Interpolator<double> interpol4(4, 0.0001);
    artnn::Interpolator<double> interpol5(5, 0.0001);

    artnn::Interpolator<double> interpol6(3, 0.00001);
    artnn::Interpolator<double> interpol7(4, 0.00001);
    artnn::Interpolator<double> interpol8(5, 0.00001);

    double sqError;

    // Network 0
    std::cout << "\nTraining Network 0 (degree = 3, etha = 0.001)\n";
    sqError = interpol0.trainFull(data, 100, 0.0);
    std::cout << "Error after 100 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol0.plotPoints(dataX, "Plots/plot0-100.csv"); 

    sqError = interpol0.trainFull(data, 900, 0.0);
    std::cout << "Error after 1000 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol0.plotPoints(dataX, "Plots/plot0-1000.csv"); 

    sqError = interpol0.trainFull(data, 9000, 0.0);
    std::cout << "Error after 10000 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol0.plotPoints(dataX, "Plots/plot0-10000.csv"); 

    // Network 1
    std::cout << "\nTraining Network 1 (degree = 4, etha = 0.001)\n";
    sqError = interpol1.trainFull(data, 100, 0.0);
    std::cout << "Error after 100 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol1.plotPoints(dataX, "Plots/plot1-100.csv"); 

    sqError = interpol1.trainFull(data, 900, 0.0);
    std::cout << "Error after 1000 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol1.plotPoints(dataX, "Plots/plot1-1000.csv"); 

    sqError = interpol1.trainFull(data, 9000, 0.0);
    std::cout << "Error after 10000 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol1.plotPoints(dataX, "Plots/plot1-10000.csv"); 

    // Network 2
    std::cout << "\nTraining Network 2 (degree = 5, etha = 0.001)\n";
    sqError = interpol2.trainFull(data, 100, 0.0);
    std::cout << "Error after 100 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol2.plotPoints(dataX, "Plots/plot2-100.csv"); 

    sqError = interpol2.trainFull(data, 900, 0.0);
    std::cout << "Error after 1000 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol2.plotPoints(dataX, "Plots/plot2-1000.csv"); 

    sqError = interpol2.trainFull(data, 9000, 0.0);
    std::cout << "Error after 10000 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol2.plotPoints(dataX, "Plots/plot2-10000.csv"); 

    // Network 3
    std::cout << "\nTraining Network 3 (degree = 3, etha = 0.0001)\n";
    sqError = interpol3.trainFull(data, 100, 0.0);
    std::cout << "Error after 100 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol3.plotPoints(dataX, "Plots/plot3-100.csv"); 

    sqError = interpol3.trainFull(data, 900, 0.0);
    std::cout << "Error after 1000 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol3.plotPoints(dataX, "Plots/plot3-1000.csv"); 

    sqError = interpol3.trainFull(data, 9000, 0.0);
    std::cout << "Error after 10000 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol3.plotPoints(dataX, "Plots/plot3-10000.csv"); 

    // Network 4
    std::cout << "\nTraining Network 4 (degree = 4, etha = 0.0001)\n";
    sqError = interpol4.trainFull(data, 100, 0.0);
    std::cout << "Error after 100 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol4.plotPoints(dataX, "Plots/plot4-100.csv"); 

    sqError = interpol4.trainFull(data, 900, 0.0);
    std::cout << "Error after 1000 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol4.plotPoints(dataX, "Plots/plot4-1000.csv"); 

    sqError = interpol4.trainFull(data, 9000, 0.0);
    std::cout << "Error after 10000 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol4.plotPoints(dataX, "Plots/plot4-10000.csv"); 

    // Network 5
    std::cout << "\nTraining Network 5 (degree = 5, etha = 0.0001)\n";
    sqError = interpol5.trainFull(data, 100, 0.0);
    std::cout << "Error after 100 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol5.plotPoints(dataX, "Plots/plot5-100.csv"); 

    sqError = interpol5.trainFull(data, 900, 0.0);
    std::cout << "Error after 1000 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol5.plotPoints(dataX, "Plots/plot5-1000.csv"); 

    sqError = interpol5.trainFull(data, 9000, 0.0);
    std::cout << "Error after 10000 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol5.plotPoints(dataX, "Plots/plot5-10000.csv"); 

    // Network 6
    std::cout << "\nTraining Network 6 (degree = 3, etha = 0.00001)\n";
    sqError = interpol6.trainFull(data, 100, 0.0);
    std::cout << "Error after 100 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol6.plotPoints(dataX, "Plots/plot6-100.csv"); 

    sqError = interpol6.trainFull(data, 900, 0.0);
    std::cout << "Error after 1000 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol6.plotPoints(dataX, "Plots/plot6-1000.csv"); 

    sqError = interpol6.trainFull(data, 9000, 0.0);
    std::cout << "Error after 10000 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol6.plotPoints(dataX, "Plots/plot6-10000.csv"); 

    // Network 7
    std::cout << "\nTraining Network 7 (degree = 4, etha = 0.00001)\n";
    sqError = interpol7.trainFull(data, 100, 0.0);
    std::cout << "Error after 100 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol7.plotPoints(dataX, "Plots/plot7-100.csv"); 

    sqError = interpol7.trainFull(data, 900, 0.0);
    std::cout << "Error after 1000 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol7.plotPoints(dataX, "Plots/plot7-1000.csv"); 

    sqError = interpol7.trainFull(data, 9000, 0.0);
    std::cout << "Error after 10000 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol7.plotPoints(dataX, "Plots/plot7-10000.csv"); 

    // Network 8
    std::cout << "\nTraining Network 8 (degree = 5, etha = 0.00001)\n";
    sqError = interpol8.trainFull(data, 100, 0.0);
    std::cout << "Error after 100 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol8.plotPoints(dataX, "Plots/plot8-100.csv"); 

    sqError = interpol8.trainFull(data, 900, 0.0);
    std::cout << "Error after 1000 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol8.plotPoints(dataX, "Plots/plot8-1000.csv"); 

    sqError = interpol8.trainFull(data, 9000, 0.0);
    std::cout << "Error after 10000 epochs: " << sqError << std::endl;
    std::cout << "Plotting points...\n";
    interpol8.plotPoints(dataX, "Plots/plot8-10000.csv"); 
}
