#include <iostream>
#include <fstream>
#include <vector>
#include <chrono> 

float convolution(const std::vector<std::vector<int>>& matrix, const std::vector<std::vector<float>>& kernel, int row, int col) {
    float result = 0.0f;
    for (int i = 0; i < kernel.size(); ++i) {
        for (int j = 0; j < kernel[i].size(); ++j) {
            result += static_cast<float>(matrix[row + i][col + j]) * kernel[i][j];
        }
    }
    return result;
}

void applyConvolution(const std::vector<std::vector<int>>& matrix, const std::vector<std::vector<float>>& kernel, std::vector<std::vector<float>>& result) {
    for (int i = 0; i < result.size(); ++i) {
        for (int j = 0; j < result[i].size(); ++j) {
            result[i][j] = convolution(matrix, kernel, i, j);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_matrix_file> <selected_kernel_file>\n";
        return 1;
    }

    std::ifstream inputFile(argv[1]);
    if (!inputFile) {
        std::cerr << "Error opening input file.\n";
        return 1;
    }

    int numRows, numCols;
    inputFile >> numRows >> numCols;

    std::vector<std::vector<int>> matrix(numRows, std::vector<int>(numCols));

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            inputFile >> matrix[i][j];
        }
    }

    std::ifstream kernelFile(argv[2]);
    if (!kernelFile) {
        std::cerr << "Error opening selected kernel file.\n";
        return 1;
    }

    int kernelSize;
    kernelFile >> kernelSize;

    std::vector<std::vector<float>> selectedKernel(kernelSize, std::vector<float>(kernelSize));

    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            kernelFile >> selectedKernel[i][j];
        }
    }

    std::vector<std::vector<float>> result(numRows - kernelSize + 1, std::vector<float>(numCols - kernelSize + 1, 0.0f));

    auto start_time = std::chrono::high_resolution_clock::now();

    applyConvolution(matrix, selectedKernel, result);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "Convolution Time: " << duration << " milliseconds\n";

    std::ofstream outputFile("serialresult.txt");
    if (!outputFile) {
        std::cerr << "Error opening output file.\n";
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
