
#include <iostream>
#include <snn.h>
#include <vector>
#include <iomanip>

#define __CHECK_DOUBLE__

typedef double DOUBLE;

template <class T>
void print_data(T* data, int n, int d) {
    std::vector<T> h_data(n * d);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            std::cout << std::setw(3) << std::setprecision(3) << h_data[i * d + j] << "   ";
        }
        std::cout << std::endl;
    }
}

#if defined( __CHECK_DOUBLE__)
int main() {

    std::cout << "This is double type check" << "\n";
    int n = 7;
    int d = 3;

    // Test SNN_DOUBLE
    std::vector<DOUBLE> data = {
        1.2, 2.0, 3.0,
        2.0, 2.4, 2.0,
        2.0, 1.0, 2.0,
        2.0, 3.2, 1.2,
        2.0, 3.1, 2.0,
        2.0, 2.2, 1.0,
        2.0, 2.1, 1.0
    };

    SNN_DOUBLE snn_index(data.data(), n, d);

    std::cout << "mean:" << std::endl;
    print_data(snn_index.d_mean, 1, d);

    std::cout << "first principal:" << std::endl;
    print_data(snn_index.d_first_pc, 1, d);

    DOUBLE R = 2.0;

    std::cout << "Single query:" << std::endl;
    std::vector<DOUBLE> new_data_unit = {2.3, 3.2, 1.0};
    std::vector<int> indices = snn_index.query_radius(new_data_unit.data(), R);

    std::cout << "Found " << indices.size() << " indices within distance " << R << ":\n";
    std::cout << "Index: ";
    for (int i = 0; i < indices.size(); i++) {
        std::cout << indices[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Multiple queries:" << std::endl;
    std::vector<DOUBLE> new_data = {
        2.3, 3.2, 1.0,
        1.5, 2.5, 2.5,
        2.1, 1.8, 1.2
    };
    int m = 3;
    std::vector<std::vector<int>> all_indices = snn_index.query_radius_batch(new_data.data(), m, R);

    for (int j = 0; j < m; j++) {
        std::cout << "Query " << j << " found " << all_indices[j].size() << " indices within distance " << R << ":\n";
        std::cout << "Index: ";
        for (int idx : all_indices[j]) {
            std::cout << idx << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

#else
    std::cout << "This is single type check" << "\n";
    int n = 7;
    int d = 3;
    std::vector<FLOAT> data = {
        1.2f, 2.0f, 3.0f,
        2.0f, 2.4f, 2.0f,
        2.0f, 1.0f, 2.0f,
        2.0f, 3.2f, 1.2f,
        2.0f, 3.1f, 2.0f,
        2.0f, 2.2f, 1.0f,
        2.0f, 2.1f, 1.0f
    };

    SNN_FLOAT snn_index(data.data(), n, d);

    std::cout << "mean:" << std::endl;
    print_data(snn_index.d_mean, 1, d);

    std::cout << "first principal:" << std::endl;
    print_data(snn_index.d_first_pc, 1, d);

    FLOAT R = 2.0f;

    std::cout << "Single query:" << std::endl;
    std::vector<FLOAT> new_data_unit = {2.3f, 3.2f, 1.0f};
    std::vector<int> indices = snn_index.query_radius(new_data_unit.data(), R);

    std::cout << "Found " << indices.size() << " indices within distance " << R << ":\n";
    std::cout << "Index: ";
    for (int i = 0; i < indices.size(); i++) {
        std::cout << indices[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Multiple queries:" << std::endl;
    std::vector<FLOAT> new_data = {
        2.3f, 3.2f, 1.0f,
        1.5f, 2.5f, 2.5f,
        2.1f, 1.8f, 1.2f
    };
    int m = 3;
    std::vector<std::vector<int>> all_indices = snn_index.query_radius_batch(new_data.data(), m, R);

    for (int j = 0; j < m; j++) {
        std::cout << "Query " << j << " found " << all_indices[j].size() << " indices within distance " << R << ":\n";
        std::cout << "Index: ";
        for (int idx : all_indices[j]) {
            std::cout << idx << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
#endif
    return 0;
}
