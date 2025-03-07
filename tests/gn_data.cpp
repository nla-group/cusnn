#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>

int main(int argc, char** argv) {
    if (argc < 2){
        std::cerr << "Please enter parameters for rows and columns!" << std::endl;
        return 0;
    }

    const int rows = atoi(argv[1]);
    const int cols = atoi(argv[2]);
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Fill matrix with random values
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dist(gen);
        }
    }

    // Write to file in row-major order
    std::ofstream file("data.txt");
    if (!file) {
        std::cerr << "Error opening fileg!\n";
        return 1;
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << matrix[i][j] << " ";
        }
        file << "\n"; // Newline after each row
    }

    file.close();
    return 0;
}
