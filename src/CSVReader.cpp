#include "CSVReader.hpp"

namespace artnn
{
    std::vector<std::vector<std::string>> CSVReader::getData()
    {
        std::ifstream file(filename);

        std::vector<std::vector<std::string>> dataList;
        std::string line = "";

        while(getline(file, line))
        {
            dataList.push_back(parseLine(line));
        }

        file.close();

        return dataList;
    }

    std::vector<std::string> CSVReader::parseLine(std::string& line)
    {
        std::vector<std::string> parsed;
        std::string currentValue = "";

        for(char c : line)
        {
            if(!isDelimiter(c))
            {
                currentValue += c;
            }
            else if(!currentValue.empty())
            {
                parsed.push_back(currentValue);
                currentValue.clear();
            }
        }

        if(!currentValue.empty())
        {
            parsed.push_back(currentValue);
            currentValue.clear();
        }

        return parsed;
    }

    bool CSVReader::isDelimiter(char c)
    {
        return delimiter.find(c) != std::string::npos;
    }
};