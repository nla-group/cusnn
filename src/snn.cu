/*
MIT License

Copyright (c) 2022 Stefan Güttel, Xinye Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "snn.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>

// #define __DEBUG__
// #define __CHECK_DOUBLE__


#if defined(__DEBUG__)
template <class T>
void print_data(T* data, int n, int d) {
    std::vector<T> h_data(n * d);
    CHECK_CUDA(cudaMemcpy(h_data.data(), data, n * d * sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            std::cout << std::setw(3) << std::setprecision(3) << h_data[i * d + j] << "   ";
        }
        std::cout << std::endl;
    }
}
#endif

// CUDA kernel definitions for FLOAT
__global__ void center_data_kernel(FLOAT* data, const FLOAT* mean, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < d) {
        data[i * d + j] -= mean[j];
    }
}

__global__ void compute_norms_kernel(FLOAT* data, FLOAT* norms, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        FLOAT norm_sq = 0.0f;
        for (int j = 0; j < d; j++) {
            FLOAT val = data[i * d + j];
            norm_sq += val * val;
        }
        norms[i] = norm_sq;
    }
}

// CUDA kernel definitions for DOUBLE
__global__ void center_data_kernel_double(DOUBLE* data, const DOUBLE* mean, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < d) {
        data[i * d + j] -= mean[j];
    }
}

__global__ void compute_norms_kernel_double(DOUBLE* data, DOUBLE* norms, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        DOUBLE norm_sq = 0.0;
        for (int j = 0; j < d; j++) {
            DOUBLE val = data[i * d + j];
            norm_sq += val * val;
        }
        norms[i] = norm_sq;
    }
}

// SNN_FLOAT implementations
SNN_FLOAT::SNN_FLOAT(FLOAT* input_data, int num_samples, int num_features)
    : n(num_samples), d(num_features), sorted_proj_idx(n) {
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    CHECK_CUDA(cudaMalloc(&d_data, n * d * sizeof(FLOAT)));
    CHECK_CUDA(cudaMalloc(&d_mean, d * sizeof(FLOAT)));
    CHECK_CUDA(cudaMalloc(&d_first_pc, d * sizeof(FLOAT)));
    CHECK_CUDA(cudaMalloc(&d_sorted_proj, n * sizeof(FLOAT)));
    CHECK_CUDA(cudaMalloc(&d_indices, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_norms, n * sizeof(FLOAT)));

    CHECK_CUDA(cudaMemcpy(d_data, input_data, n * d * sizeof(FLOAT), cudaMemcpyHostToDevice));
    compute_first_pc();
}

SNN_FLOAT::~SNN_FLOAT() {
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_mean));
    CHECK_CUDA(cudaFree(d_first_pc));
    CHECK_CUDA(cudaFree(d_sorted_proj));
    CHECK_CUDA(cudaFree(d_indices));
    CHECK_CUDA(cudaFree(d_norms));
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
}

std::vector<FLOAT> SNN_FLOAT::get_first_pc() const {
    std::vector<FLOAT> h_first_pc(d);
    CHECK_CUDA(cudaMemcpy(h_first_pc.data(), d_first_pc, d * sizeof(FLOAT), cudaMemcpyDeviceToHost));
    return h_first_pc;
}

void SNN_FLOAT::compute_first_pc() {
    std::vector<FLOAT> h_mean(d, 0.0f);
    for (int j = 0; j < d; j++) {
        float sum;
        CHECK_CUBLAS(cublasSasum(cublas_handle, n, d_data + j, d, &sum));
        h_mean[j] = sum / n;
    }
    CHECK_CUDA(cudaMemcpy(d_mean, h_mean.data(), d * sizeof(FLOAT), cudaMemcpyHostToDevice));

    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x, (d + threads.y - 1) / threads.y);
    center_data_kernel<<<blocks, threads>>>(d_data, d_mean, n, d);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<FLOAT> dis(-1.0f, 1.0f);
    std::vector<FLOAT> h_first_pc(d);
    for (int i = 0; i < d; i++) h_first_pc[i] = dis(gen);
    CHECK_CUDA(cudaMemcpy(d_first_pc, h_first_pc.data(), d * sizeof(FLOAT), cudaMemcpyHostToDevice));

    FLOAT* d_temp;
    CHECK_CUDA(cudaMalloc(&d_temp, n * sizeof(FLOAT)));
    CHECK_CUDA(cudaMemset(d_temp, 0, n * sizeof(FLOAT)));
    float alpha = 1.0f, beta = 0.0f;
    const int max_iter = 100;
    const FLOAT tol = 1e-6f;
    FLOAT norm_prev = 0.0f;

    CHECK_CUDA(cudaDeviceSynchronize());

    for (int iter = 0; iter < max_iter; iter++) {
        CHECK_CUBLAS(cublasSgemv(cublas_handle, CUBLAS_OP_T, d, n, &alpha, d_data, d,
                                 d_first_pc, 1, &beta, d_temp, 1));
        CHECK_CUBLAS(cublasSgemv(cublas_handle, CUBLAS_OP_N, d, n, &alpha, d_data, d,
                                 d_temp, 1, &beta, d_first_pc, 1));

        FLOAT norm;
        CHECK_CUBLAS(cublasSnrm2(cublas_handle, d, d_first_pc, 1, &norm));
        if (norm < 1e-10f) {
            std::cerr << "Zero norm encountered\n";
            break;
        }
        float scale = alpha / norm;
        CHECK_CUBLAS(cublasSscal(cublas_handle, d, &scale, d_first_pc, 1));

        if (iter > 0 && std::abs(norm - norm_prev) < tol * norm) {
            break;
        }
        norm_prev = norm;
    }

    FLOAT* d_projections;
    CHECK_CUDA(cudaMalloc(&d_projections, n * sizeof(FLOAT)));
    compute_projections_and_norms(d_projections);
    CHECK_CUDA(cudaFree(d_temp));
    CHECK_CUDA(cudaFree(d_projections));
}

void SNN_FLOAT::compute_projections_and_norms(FLOAT* d_projections) {
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemv(cublas_handle, CUBLAS_OP_T, d, n, &alpha, d_data, d,
                             d_first_pc, 1, &beta, d_projections, 1));

    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    compute_norms_kernel<<<blocks, threads_per_block>>>(d_data, d_norms, n, d);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<FLOAT> h_projections(n);
    std::vector<int> h_indices(n);
    std::vector<FLOAT> h_norms(n);
    CHECK_CUDA(cudaMemcpy(h_projections.data(), d_projections, n * sizeof(FLOAT), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_norms.data(), d_norms, n * sizeof(FLOAT), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) {
        sorted_proj_idx[i] = std::make_tuple(h_projections[i], i, h_norms[i]);
    }
    std::sort(sorted_proj_idx.begin(), sorted_proj_idx.end(),
              [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });

    for (int i = 0; i < n; i++) {
        h_projections[i] = std::get<0>(sorted_proj_idx[i]);
        h_indices[i] = std::get<1>(sorted_proj_idx[i]);
        h_norms[i] = std::get<2>(sorted_proj_idx[i]);
    }
    CHECK_CUDA(cudaMemcpy(d_sorted_proj, h_projections.data(), n * sizeof(FLOAT), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_indices, h_indices.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_norms, h_norms.data(), n * sizeof(FLOAT), cudaMemcpyHostToDevice));
}

std::vector<int> SNN_FLOAT::query_radius(const FLOAT* new_data, FLOAT R) const {
    FLOAT R_sq = R * R;

    FLOAT* d_centered;
    CHECK_CUDA(cudaMalloc(&d_centered, d * sizeof(FLOAT)));
    std::vector<FLOAT> h_centered(d);
    for (int j = 0; j < d; j++) h_centered[j] = new_data[j];
    CHECK_CUDA(cudaMemcpy(d_centered, h_centered.data(), d * sizeof(FLOAT), cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int blocks = (d + threads_per_block - 1) / threads_per_block;
    center_data_kernel<<<blocks, threads_per_block>>>(d_centered, d_mean, 1, d);
    CHECK_CUDA(cudaDeviceSynchronize());

    FLOAT q;
    CHECK_CUBLAS(cublasSdot(cublas_handle, d, d_first_pc, 1, d_centered, 1, &q));
    FLOAT new_norm_sq;
    CHECK_CUBLAS(cublasSdot(cublas_handle, d, d_centered, 1, d_centered, 1, &new_norm_sq));

    FLOAT* d_dot_products;
    CHECK_CUDA(cudaMalloc(&d_dot_products, n * sizeof(FLOAT)));
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemv(cublas_handle, CUBLAS_OP_T, d, n, &alpha, d_data, d,
                             d_centered, 1, &beta, d_dot_products, 1));

    std::vector<FLOAT> h_dot_products(n);
    CHECK_CUDA(cudaMemcpy(h_dot_products.data(), d_dot_products, n * sizeof(FLOAT), cudaMemcpyDeviceToHost));

    auto lower_it = std::lower_bound(sorted_proj_idx.begin(), sorted_proj_idx.end(),
                                     q - R,
                                     [](const auto& p, FLOAT val) { return std::get<0>(p) < val; });
    auto upper_it = std::upper_bound(sorted_proj_idx.begin(), sorted_proj_idx.end(),
                                     q + R,
                                     [](FLOAT val, const auto& p) { return val < std::get<0>(p); });

    std::vector<int> indices;
    indices.reserve(upper_it - lower_it);
    for (auto it = lower_it; it != upper_it; ++it) {
        int idx = std::get<1>(*it);
        FLOAT dot_xy = h_dot_products[idx];
        FLOAT norm_sq = std::get<2>(*it);
        FLOAT dist_sq = norm_sq + new_norm_sq - 2.0f * dot_xy;
        if (dist_sq <= R_sq) indices.push_back(idx);
    }

    CHECK_CUDA(cudaFree(d_centered));
    CHECK_CUDA(cudaFree(d_dot_products));
    return indices;
}

std::vector<std::vector<int>> SNN_FLOAT::query_radius_batch(const FLOAT* new_data, int m, FLOAT R) const {
    FLOAT R_sq = R * R;

    FLOAT* d_centered;
    CHECK_CUDA(cudaMalloc(&d_centered, m * d * sizeof(FLOAT)));
    CHECK_CUDA(cudaMemcpy(d_centered, new_data, m * d * sizeof(FLOAT), cudaMemcpyHostToDevice));
    dim3 threads(16, 16);
    dim3 blocks((m + threads.x - 1) / threads.x, (d + threads.y - 1) / threads.y);
    center_data_kernel<<<blocks, threads>>>(d_centered, d_mean, m, d);
    CHECK_CUDA(cudaDeviceSynchronize());

    FLOAT* d_q_values;
    CHECK_CUDA(cudaMalloc(&d_q_values, m * sizeof(FLOAT)));
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemv(cublas_handle, CUBLAS_OP_T, d, m, &alpha, d_centered, d,
                             d_first_pc, 1, &beta, d_q_values, 1));

    FLOAT* d_new_norm_sq;
    CHECK_CUDA(cudaMalloc(&d_new_norm_sq, m * sizeof(FLOAT)));
    int threads_per_block = 256;
    int norm_blocks = (m + threads_per_block - 1) / threads_per_block;
    compute_norms_kernel<<<norm_blocks, threads_per_block>>>(d_centered, d_new_norm_sq, m, d);
    CHECK_CUDA(cudaDeviceSynchronize());

    FLOAT* d_dot_products;
    CHECK_CUDA(cudaMalloc(&d_dot_products, n * m * sizeof(FLOAT)));
    CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, d, &alpha,
                             d_data, d, d_centered, d, &beta, d_dot_products, n));

    std::vector<FLOAT> h_q_values(m);
    std::vector<FLOAT> h_new_norm_sq(m);
    std::vector<FLOAT> h_dot_products(n * m);
    CHECK_CUDA(cudaMemcpy(h_q_values.data(), d_q_values, m * sizeof(FLOAT), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_new_norm_sq.data(), d_new_norm_sq, m * sizeof(FLOAT), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_dot_products.data(), d_dot_products, n * m * sizeof(FLOAT), cudaMemcpyDeviceToHost));

    std::vector<std::vector<int>> all_indices(m);
    for (int j = 0; j < m; j++) {
        FLOAT q = h_q_values[j];
        auto lower_it = std::lower_bound(sorted_proj_idx.begin(), sorted_proj_idx.end(),
                                         q - R,
                                         [](const auto& p, FLOAT val) { return std::get<0>(p) < val; });
        auto upper_it = std::upper_bound(sorted_proj_idx.begin(), sorted_proj_idx.end(),
                                         q + R,
                                         [](FLOAT val, const auto& p) { return val < std::get<0>(p); });

        std::vector<int>& indices = all_indices[j];
        indices.reserve(upper_it - lower_it);

        for (auto it = lower_it; it != upper_it; ++it) {
            int idx = std::get<1>(*it);
            FLOAT norm_sq = std::get<2>(*it);
            FLOAT dot_xy = h_dot_products[idx + j * n];
            FLOAT dist_sq = norm_sq + h_new_norm_sq[j] - 2.0f * dot_xy;
            if (dist_sq <= R_sq) indices.push_back(idx);
        }
    }

    CHECK_CUDA(cudaFree(d_centered));
    CHECK_CUDA(cudaFree(d_q_values));
    CHECK_CUDA(cudaFree(d_new_norm_sq));
    CHECK_CUDA(cudaFree(d_dot_products));
    return all_indices;
}

// SNN_DOUBLE implementations
SNN_DOUBLE::SNN_DOUBLE(DOUBLE* input_data, int num_samples, int num_features)
    : n(num_samples), d(num_features), sorted_proj_idx(n) {
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    CHECK_CUDA(cudaMalloc(&d_data, n * d * sizeof(DOUBLE)));
    CHECK_CUDA(cudaMalloc(&d_mean, d * sizeof(DOUBLE)));
    CHECK_CUDA(cudaMalloc(&d_first_pc, d * sizeof(DOUBLE)));
    CHECK_CUDA(cudaMalloc(&d_sorted_proj, n * sizeof(DOUBLE)));
    CHECK_CUDA(cudaMalloc(&d_indices, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_norms, n * sizeof(DOUBLE)));

    CHECK_CUDA(cudaMemcpy(d_data, input_data, n * d * sizeof(DOUBLE), cudaMemcpyHostToDevice));
    compute_first_pc();
}

SNN_DOUBLE::~SNN_DOUBLE() {
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_mean));
    CHECK_CUDA(cudaFree(d_first_pc));
    CHECK_CUDA(cudaFree(d_sorted_proj));
    CHECK_CUDA(cudaFree(d_indices));
    CHECK_CUDA(cudaFree(d_norms));
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
}

std::vector<DOUBLE> SNN_DOUBLE::get_first_pc() const {
    std::vector<DOUBLE> h_first_pc(d);
    CHECK_CUDA(cudaMemcpy(h_first_pc.data(), d_first_pc, d * sizeof(DOUBLE), cudaMemcpyDeviceToHost));
    return h_first_pc;
}

void SNN_DOUBLE::compute_first_pc() {
    std::vector<DOUBLE> h_mean(d, 0.0);
    for (int j = 0; j < d; j++) {
        double sum;
        CHECK_CUBLAS(cublasDasum(cublas_handle, n, d_data + j, d, &sum));
        h_mean[j] = sum / n;
    }
    CHECK_CUDA(cudaMemcpy(d_mean, h_mean.data(), d * sizeof(DOUBLE), cudaMemcpyHostToDevice));

    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x, (d + threads.y - 1) / threads.y);
    center_data_kernel_double<<<blocks, threads>>>(d_data, d_mean, n, d);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<DOUBLE> dis(-1.0, 1.0);
    std::vector<DOUBLE> h_first_pc(d);
    for (int i = 0; i < d; i++) h_first_pc[i] = dis(gen);
    CHECK_CUDA(cudaMemcpy(d_first_pc, h_first_pc.data(), d * sizeof(DOUBLE), cudaMemcpyHostToDevice));

    DOUBLE* d_temp;
    CHECK_CUDA(cudaMalloc(&d_temp, n * sizeof(DOUBLE)));
    CHECK_CUDA(cudaMemset(d_temp, 0, n * sizeof(DOUBLE)));
    double alpha = 1.0, beta = 0.0;
    const int max_iter = 100;
    const DOUBLE tol = 1e-10;
    DOUBLE norm_prev = 0.0;

    CHECK_CUDA(cudaDeviceSynchronize());

    for (int iter = 0; iter < max_iter; iter++) {
        CHECK_CUBLAS(cublasDgemv(cublas_handle, CUBLAS_OP_T, d, n, &alpha, d_data, d,
                                 d_first_pc, 1, &beta, d_temp, 1));
        CHECK_CUBLAS(cublasDgemv(cublas_handle, CUBLAS_OP_N, d, n, &alpha, d_data, d,
                                 d_temp, 1, &beta, d_first_pc, 1));

        DOUBLE norm;
        CHECK_CUBLAS(cublasDnrm2(cublas_handle, d, d_first_pc, 1, &norm));
        if (norm < 1e-10) {
            std::cerr << "Zero norm encountered\n";
            break;
        }
        double scale = alpha / norm;
        CHECK_CUBLAS(cublasDscal(cublas_handle, d, &scale, d_first_pc, 1));

        if (iter > 0 && std::abs(norm - norm_prev) < tol * norm) {
            break;
        }
        norm_prev = norm;
    }

    DOUBLE* d_projections;
    CHECK_CUDA(cudaMalloc(&d_projections, n * sizeof(DOUBLE)));
    compute_projections_and_norms(d_projections);
    CHECK_CUDA(cudaFree(d_temp));
    CHECK_CUDA(cudaFree(d_projections));
}

void SNN_DOUBLE::compute_projections_and_norms(DOUBLE* d_projections) {
    double alpha = 1.0, beta = 0.0;
    CHECK_CUBLAS(cublasDgemv(cublas_handle, CUBLAS_OP_T, d, n, &alpha, d_data, d,
                             d_first_pc, 1, &beta, d_projections, 1));

    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    compute_norms_kernel_double<<<blocks, threads_per_block>>>(d_data, d_norms, n, d);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<DOUBLE> h_projections(n);
    std::vector<int> h_indices(n);
    std::vector<DOUBLE> h_norms(n);
    CHECK_CUDA(cudaMemcpy(h_projections.data(), d_projections, n * sizeof(DOUBLE), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_norms.data(), d_norms, n * sizeof(DOUBLE), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) {
        sorted_proj_idx[i] = std::make_tuple(h_projections[i], i, h_norms[i]);
    }
    std::sort(sorted_proj_idx.begin(), sorted_proj_idx.end(),
              [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });

    for (int i = 0; i < n; i++) {
        h_projections[i] = std::get<0>(sorted_proj_idx[i]);
        h_indices[i] = std::get<1>(sorted_proj_idx[i]);
        h_norms[i] = std::get<2>(sorted_proj_idx[i]);
    }
    CHECK_CUDA(cudaMemcpy(d_sorted_proj, h_projections.data(), n * sizeof(DOUBLE), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_indices, h_indices.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_norms, h_norms.data(), n * sizeof(DOUBLE), cudaMemcpyHostToDevice));
}

std::vector<int> SNN_DOUBLE::query_radius(const DOUBLE* new_data, DOUBLE R) const {
    DOUBLE R_sq = R * R;

    DOUBLE* d_centered;
    CHECK_CUDA(cudaMalloc(&d_centered, d * sizeof(DOUBLE)));
    std::vector<DOUBLE> h_centered(d);
    for (int j = 0; j < d; j++) h_centered[j] = new_data[j];
    CHECK_CUDA(cudaMemcpy(d_centered, h_centered.data(), d * sizeof(DOUBLE), cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int blocks = (d + threads_per_block - 1) / threads_per_block;
    center_data_kernel_double<<<blocks, threads_per_block>>>(d_centered, d_mean, 1, d);
    CHECK_CUDA(cudaDeviceSynchronize());

    DOUBLE q;
    CHECK_CUBLAS(cublasDdot(cublas_handle, d, d_first_pc, 1, d_centered, 1, &q));
    DOUBLE new_norm_sq;
    CHECK_CUBLAS(cublasDdot(cublas_handle, d, d_centered, 1, d_centered, 1, &new_norm_sq));

    DOUBLE* d_dot_products;
    CHECK_CUDA(cudaMalloc(&d_dot_products, n * sizeof(DOUBLE)));
    double alpha = 1.0, beta = 0.0;
    CHECK_CUBLAS(cublasDgemv(cublas_handle, CUBLAS_OP_T, d, n, &alpha, d_data, d,
                             d_centered, 1, &beta, d_dot_products, 1));

    std::vector<DOUBLE> h_dot_products(n);
    CHECK_CUDA(cudaMemcpy(h_dot_products.data(), d_dot_products, n * sizeof(DOUBLE), cudaMemcpyDeviceToHost));

    auto lower_it = std::lower_bound(sorted_proj_idx.begin(), sorted_proj_idx.end(),
                                     q - R,
                                     [](const auto& p, DOUBLE val) { return std::get<0>(p) < val; });
    auto upper_it = std::upper_bound(sorted_proj_idx.begin(), sorted_proj_idx.end(),
                                     q + R,
                                     [](DOUBLE val, const auto& p) { return val < std::get<0>(p); });

    std::vector<int> indices;
    indices.reserve(upper_it - lower_it);
    for (auto it = lower_it; it != upper_it; ++it) {
        int idx = std::get<1>(*it);
        DOUBLE dot_xy = h_dot_products[idx];
        DOUBLE norm_sq = std::get<2>(*it);
        DOUBLE dist_sq = norm_sq + new_norm_sq - 2.0 * dot_xy;
        if (dist_sq <= R_sq) indices.push_back(idx);
    }

    CHECK_CUDA(cudaFree(d_centered));
    CHECK_CUDA(cudaFree(d_dot_products));
    return indices;
}

std::vector<std::vector<int>> SNN_DOUBLE::query_radius_batch(const DOUBLE* new_data, int m, DOUBLE R) const {
    DOUBLE R_sq = R * R;

    DOUBLE* d_centered;
    CHECK_CUDA(cudaMalloc(&d_centered, m * d * sizeof(DOUBLE)));
    CHECK_CUDA(cudaMemcpy(d_centered, new_data, m * d * sizeof(DOUBLE), cudaMemcpyHostToDevice));
    dim3 threads(16, 16);
    dim3 blocks((m + threads.x - 1) / threads.x, (d + threads.y - 1) / threads.y);
    center_data_kernel_double<<<blocks, threads>>>(d_centered, d_mean, m, d);
    CHECK_CUDA(cudaDeviceSynchronize());

    DOUBLE* d_q_values;
    CHECK_CUDA(cudaMalloc(&d_q_values, m * sizeof(DOUBLE)));
    double alpha = 1.0, beta = 0.0;
    CHECK_CUBLAS(cublasDgemv(cublas_handle, CUBLAS_OP_T, d, m, &alpha, d_centered, d,
                             d_first_pc, 1, &beta, d_q_values, 1));

    DOUBLE* d_new_norm_sq;
    CHECK_CUDA(cudaMalloc(&d_new_norm_sq, m * sizeof(DOUBLE)));
    int threads_per_block = 256;
    int norm_blocks = (m + threads_per_block - 1) / threads_per_block;
    compute_norms_kernel_double<<<norm_blocks, threads_per_block>>>(d_centered, d_new_norm_sq, m, d);
    CHECK_CUDA(cudaDeviceSynchronize());

    DOUBLE* d_dot_products;
    CHECK_CUDA(cudaMalloc(&d_dot_products, n * m * sizeof(DOUBLE)));
    CHECK_CUBLAS(cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, d, &alpha,
                             d_data, d, d_centered, d, &beta, d_dot_products, n));

    std::vector<DOUBLE> h_q_values(m);
    std::vector<DOUBLE> h_new_norm_sq(m);
    std::vector<DOUBLE> h_dot_products(n * m);
    CHECK_CUDA(cudaMemcpy(h_q_values.data(), d_q_values, m * sizeof(DOUBLE), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_new_norm_sq.data(), d_new_norm_sq, m * sizeof(DOUBLE), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_dot_products.data(), d_dot_products, n * m * sizeof(DOUBLE), cudaMemcpyDeviceToHost));

    std::vector<std::vector<int>> all_indices(m);
    for (int j = 0; j < m; j++) {
        DOUBLE q = h_q_values[j];
        auto lower_it = std::lower_bound(sorted_proj_idx.begin(), sorted_proj_idx.end(),
                                         q - R,
                                         [](const auto& p, DOUBLE val) { return std::get<0>(p) < val; });
        auto upper_it = std::upper_bound(sorted_proj_idx.begin(), sorted_proj_idx.end(),
                                         q + R,
                                         [](DOUBLE val, const auto& p) { return val < std::get<0>(p); });

        std::vector<int>& indices = all_indices[j];
        indices.reserve(upper_it - lower_it);

        for (auto it = lower_it; it != upper_it; ++it) {
            int idx = std::get<1>(*it);
            DOUBLE norm_sq = std::get<2>(*it);
            DOUBLE dot_xy = h_dot_products[idx + j * n];
            DOUBLE dist_sq = norm_sq + h_new_norm_sq[j] - 2.0 * dot_xy;
            if (dist_sq <= R_sq) indices.push_back(idx);
        }
    }

    CHECK_CUDA(cudaFree(d_centered));
    CHECK_CUDA(cudaFree(d_q_values));
    CHECK_CUDA(cudaFree(d_new_norm_sq));
    CHECK_CUDA(cudaFree(d_dot_products));
    return all_indices;
}
