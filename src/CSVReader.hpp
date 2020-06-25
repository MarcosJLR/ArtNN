/*
 * CSV reader
 * 
 * Class to abstract reading from a csv file
 *
 * Author: Marcos Lerones
 */

#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

namespace artnn
{
    class CSVReader
    {
    public:
        CSVReader(std::string filename, std::string delim = ",")
            : filename(filename), delimiter(delim) {}

        // Get the contents of the file
        std::vector<std::vector<std::string>> getData();

    private:
        std::string filename;
        std::string delimiter;

        // Helper Functions
        std::vector<std::string> parseLine(std::string& line);
        bool isDelimiter(char c);
    };
};