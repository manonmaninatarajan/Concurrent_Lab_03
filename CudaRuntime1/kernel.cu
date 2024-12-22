#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "device_launch_parameters.h"

#define BLOCK_SIZE 256
#define MAX_TEXT_LENGTH 50 

// Struct for input data
struct mydata {
    char text[MAX_TEXT_LENGTH]; 
    int number;
    double value;
};

// Struct for results
struct Result {
    char data[80];  // Result string length, can be adjusted as needed
};

// Function to read data from the file
std::vector<mydata> readFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<mydata> items;
    std::string line;
    while (std::getline(file, line)) {
        mydata item;
        size_t pos1 = line.find(';');
        size_t pos2 = line.find(';', pos1 + 1);

        // Ensure there are valid semicolons in the line
        if (pos1 != std::string::npos && pos2 != std::string::npos) {
            // Parse text, number, and value
            line.copy(item.text, pos1);  // Copy text before the first semicolon
            item.text[pos1] = '\0';  // Null-terminate the string

            item.number = std::stoi(line.substr(pos1 + 1, pos2 - pos1 - 1));
            item.value = std::stod(line.substr(pos2 + 1));
            items.push_back(item);
        }
        else {
            std::cerr << "Skipping invalid line: " << line << std::endl;
        }
    }

    file.close();
    return items;
}

 __global__ void filterAndComputeItems(mydata * items, int size, Result * results) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        mydata item = items[index];

        // Check if the number is even
        if (item.number % 2 != 0) {
            // Set result to an empty string for odd numbers (no output)
            results[index].data[0] = '\0';  // Null-terminate the result string immediately
            return;
        }

        int pos = 0;

        // Copy the text (null-terminated) into the result
        for (int i = 0; item.text[i] != '\0'; i++) {
            results[index].data[pos++] = item.text[i];
        }

        // Add the '-' separator
        results[index].data[pos++] = '-';

        // Convert the number to string and append it
        int tempNumber = item.number;
        char numberStr[12];  // Buffer to hold the number as a string
        int numLen = 0;

        do {
            numberStr[numLen++] = '0' + (tempNumber % 10);
            tempNumber /= 10;
        } while (tempNumber > 0);

        // Reverse the number string and append it
        for (int i = numLen - 1; i >= 0; i--) {
            results[index].data[pos++] = numberStr[i];
        }

        // Add the '-' separator
        results[index].data[pos++] = '-';

        // Convert the value to string and append it
        char valueStr[20];  // Buffer to hold the value as a string
        int len = 0;
        float val = (float)item.value;
        int intPart = (int)val;
        float fracPart = val - intPart;

        // Convert integer part
        do {
            valueStr[len++] = '0' + (intPart % 10);
            intPart /= 10;
        } while (intPart > 0);

        // Reverse the integer part
        for (int i = 0; i < len / 2; i++) {
            char temp = valueStr[i];
            valueStr[i] = valueStr[len - i - 1];
            valueStr[len - i - 1] = temp;
        }

        // Add decimal point and fractional part
        valueStr[len++] = '.';
        fracPart *= 100;
        int fracInt = (int)fracPart;
        valueStr[len++] = '0' + (fracInt / 10);
        valueStr[len++] = '0' + (fracInt % 10);

        // Append the value
        for (int i = 0; i < len; i++) {
            results[index].data[pos++] = valueStr[i];
        }

        // Compute the result if the number is even
        double computedResult = 0.0;
        computedResult = item.number * item.value;  // Multiply number and value

        // Add the computed result to the result string
        results[index].data[pos++] = ';';  // Add a semicolon

        // Convert the computed result to string
        valueStr[0] = '\0';  // Reset value string
        len = 0;
        int computedInt = (int)computedResult;
        float computedFrac = computedResult - computedInt;

        // Convert integer part of computed result
        do {
            valueStr[len++] = '0' + (computedInt % 10);
            computedInt /= 10;
        } while (computedInt > 0);

        // Reverse the integer part
        for (int i = 0; i < len / 2; i++) {
            char temp = valueStr[i];
            valueStr[i] = valueStr[len - i - 1];
            valueStr[len - i - 1] = temp;
        }

        // Add decimal point and fractional part for computed result
        valueStr[len++] = '.';
        computedFrac *= 100;
        int fracIntRes = (int)computedFrac;
        valueStr[len++] = '0' + (fracIntRes / 10);
        valueStr[len++] = '0' + (fracIntRes % 10);

        // Append the computed result
        for (int i = 0; i < len; i++) {
            results[index].data[pos++] = valueStr[i];
        }

        // Null-terminate the result string
        results[index].data[pos] = '\0';
    }
}

// Function to count the number of lines (records) in the input file
int countInputDataRecords(const std::string& filename) {
    std::ifstream file(filename);
    int count = 0;
    std::string line;

    // Count each line in the file
    while (std::getline(file, line)) {
        count++;
    }

    file.close();
    return count;
}

// Function to write results to a file
void writeResultsToFile(const Result* results, int size, const std::string& filename, int totalInputData) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    int resultCount = 0;

    for (int i = 0; i < size; i++) {
        if (results[i].data[0] != '\0') {  // Check if the result is not empty
            file << results[i].data << std::endl;
            resultCount++;
        }
    }

    file.close();
    std::cout << "Successfully wrote results to file: " << filename << std::endl;
    std::cout << "Total number of input data in mydata.txt: " << totalInputData << std::endl;  // Print the total number of records in the input file
    std::cout << "Total number of items in result file: " << resultCount << std::endl;  // Print the number of valid results written
}


int main() {
    // File selection
    std::string inputFile;
    std::cout << "Enter input file name: ";
    std::cin >> inputFile;

    // Count the total number of records in the input file
    int totalInputData = countInputDataRecords(inputFile);

    // Now read the input file
    std::vector<mydata> hostItems = readFile(inputFile);  // Your existing function to read the file
    const int dataSize = hostItems.size();

    // allocating memory for host and device
    mydata* deviceItems;
    Result* deviceResults;
    Result* hostResults = new Result[dataSize];

    cudaMalloc(&deviceItems, dataSize * sizeof(mydata));
    cudaMalloc(&deviceResults, dataSize * sizeof(Result));

    cudaMemcpy(deviceItems, hostItems.data(), dataSize * sizeof(mydata), cudaMemcpyHostToDevice);//from cpu to gpu

    // Launch kernel
    int blocks = (dataSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    filterAndComputeItems << <blocks, BLOCK_SIZE >> > (deviceItems, dataSize, deviceResults);

    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();//wait till the gpu runs all the task excecuting threads

    cudaMemcpy(hostResults, deviceResults, dataSize * sizeof(Result), cudaMemcpyDeviceToHost);//from gpu to cpu

    // Write results to the file, passing the total input data count
    std::string outputFile = "results.txt";
    writeResultsToFile(hostResults, dataSize, outputFile, totalInputData);  

    // Clean up memory
    delete[] hostResults;
    cudaFree(deviceItems);
    cudaFree(deviceResults);

    return 0;
}


