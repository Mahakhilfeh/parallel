#include <iostream>
#include <fstream>
#include <vector>
#include <chrono> 
#include <omp.h>

void loadKernelFromFile(const std::string& kernelFileName, std::vector<std::vector<float>>& kernel);

float convolution(const std::vector<std::vector<float>>& matrix, const std::vector<std::vector<float>>& kernel, int row, int col) {
    float result = 0.0f;
    for (int i = 0; i < kernel.size(); ++i) {
        for (int j = 0; j < kernel[i].size(); ++j) {
            result += matrix[row + i][col + j] * kernel[i][j];
        }
    }
    return result;
}

void applyConvolutionParallel(const std::vector<std::vector<float>>& matrix, const std::vector<std::vector<float>>& kernel, std::vector<std::vector<float>>& result, int numThreads) {
    int numRows = result.size();
    int numCols = (numRows > 0) ? result[0].size() : 0;  

    int rowsPerThread = numRows / numThreads;

    #pragma omp parallel num_threads(numThreads)
    {
        #pragma omp for
        for (int i = 0; i < numThreads; ++i) {
            int startRow = i * rowsPerThread;
            int endRow = (i == numThreads - 1) ? numRows : (i + 1) * rowsPerThread;
            for (int row = startRow; row < endRow; ++row) {
                for (int col = 0; col < numCols; ++col) {
                    result[row][col] = convolution(matrix, kernel, row, col);
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <num_threads> <kernel_type> <kernel_file>\n";
        return 1;
    }

    int numThreads = std::stoi(argv[1]);
    if (numThreads <= 0) {
        std::cout << "Number of threads should be greater than 0.\n";
        return 1;
    }

    std::string kernelType = argv[2];
    std::string kernelFileName = argv[3];

    std::vector<std::vector<float>> kernel;
    loadKernelFromFile(kernelFileName, kernel);

    std::ifstream inputFile("oimage.txt");
    if (!inputFile) {
        std::cout << "Error opening input file.\n";
        return 1;
    }

    int numRows, numCols;
    inputFile >> numRows >> numCols;

    if (numRows <= 0 || numCols <= 0) {
        std::cout << "Invalid matrix size in the input file.\n";
        return 1;
    }

    std::vector<std::vector<float>> matrix(numRows, std::vector<float>(numCols));

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            inputFile >> matrix[i][j];
        }
    }

    std::vector<std::vector<float>> result(numRows - kernel.size() + 1, std::vector<float>(numCols - kernel[0].size() + 1, 0.0f));

    auto start_time = std::chrono::high_resolution_clock::now();

    applyConvolutionParallel(matrix, kernel, result, numThreads);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "execution Time: " << duration << " milliseconds\n";

    std::ofstream outputFile("parallelresult.txt");
    if (!outputFile) {
        std::cerr << "Error opening output file for writing." << std::endl;
        return 1;
    }

    for (const auto& row : result) {
        for (float value : row) {
            outputFile << value << '\t';
        }
        outputFile << '\n';
    }

    return 0;
}

void loadKernelFromFile(const std::string& kernelFileName, std::vector<std::vector<float>>& kernel) {
    std::ifstream kernelFile(kernelFileName);
    if (!kernelFile) {
        std::cerr << "Error opening kernel file: " << kernelFileName << std::endl;
        exit(1);
    }

    int kernelSize;
    kernelFile >> kernelSize;

    kernel.resize(kernelSize, std::vector<float>(kernelSize));

    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            kernelFile >> kernel[i][j];
        }
    }
}
