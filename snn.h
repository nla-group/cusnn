#ifndef SNN_H
#define SNN_H

#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

typedef float FLOAT;
typedef double DOUBLE;

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t stat = call; \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error: " << stat << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA kernels for FLOAT
__global__ void center_data_kernel(FLOAT* data, const FLOAT* mean, int n, int d);
__global__ void compute_norms_kernel(FLOAT* data, FLOAT* norms, int n, int d);

// CUDA kernels for DOUBLE
__global__ void center_data_kernel_double(DOUBLE* data, const DOUBLE* mean, int n, int d);
__global__ void compute_norms_kernel_double(DOUBLE* data, DOUBLE* norms, int n, int d);

class SNN_FLOAT {
private:
    int n; // Number of samples
    int d; // Number of features
    FLOAT* d_data; // Device-centered data (n x d, row-major)
    FLOAT* d_sorted_proj; // Device projections
    int* d_indices; // Device indices
    FLOAT* d_norms; // Device squared norms
    std::vector<std::tuple<FLOAT, int, FLOAT>> sorted_proj_idx; // Host sorted data
    cublasHandle_t cublas_handle;

    void compute_projections_and_norms(FLOAT* d_projections);

public:
    FLOAT* d_mean; // Device feature means (d)
    FLOAT* d_first_pc; // Device first principal component (d x 1)

    SNN_FLOAT(FLOAT* input_data, int num_samples, int num_features);
    ~SNN_FLOAT();

    std::vector<FLOAT> get_first_pc() const;
    std::vector<int> query_radius(const FLOAT* new_data, FLOAT R) const;
    std::vector<std::vector<int>> query_radius_batch(const FLOAT* new_data, int m, FLOAT R) const;

private:
    void compute_first_pc();
};

class SNN_DOUBLE {
private:
    int n; // Number of samples
    int d; // Number of features
    DOUBLE* d_data; // Device-centered data (n x d, row-major)
    DOUBLE* d_sorted_proj; // Device projections
    int* d_indices; // Device indices
    DOUBLE* d_norms; // Device squared norms
    std::vector<std::tuple<DOUBLE, int, DOUBLE>> sorted_proj_idx; // Host sorted data
    cublasHandle_t cublas_handle;

    void compute_projections_and_norms(DOUBLE* d_projections);

public:
    DOUBLE* d_mean; // Device feature means (d)
    DOUBLE* d_first_pc; // Device first principal component (d x 1)

    SNN_DOUBLE(DOUBLE* input_data, int num_samples, int num_features);
    ~SNN_DOUBLE();

    std::vector<DOUBLE> get_first_pc() const;
    std::vector<int> query_radius(const DOUBLE* new_data, DOUBLE R) const;
    std::vector<std::vector<int>> query_radius_batch(const DOUBLE* new_data, int m, DOUBLE R) const;

private:
    void compute_first_pc();
};

#endif // SNN_H