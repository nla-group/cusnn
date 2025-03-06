# CUSNN - CUDA-based Fast Exact Nearest Neighbor Search Algorithm

CUSNN (CUDA-based SNN Exact Nearest Neighbor Search) is a high-performance algorithm designed to perform fast and precise nearest neighbor searches on large datasets. By leveraging the full power of CUDA, CUSNN accelerates the search process, utilizing search space pruning, efficient memory management and advanced optimization techniques. This makes it ideal for applications where both accuracy and speed are critical, such as in large-scale data analysis and high-dimensional search tasks.

## Key Features

- **Exact Nearest Neighbor Search**: Unlike approximate search methods, CUSNN ensures that the results are exact, making it suitable for applications that require high precision and reliability.
  
- **CUDA Optimization**: CUSNN takes of GPU parallelism, significantly speeding up the nearest neighbor search process. The algorithm is designed to run efficiently on modern CUDA-capable hardware, ensuring high throughput even with large datasets.

- **Continuous Memory Indexing**: CUSNN uses continuous memory indexing to optimize data access patterns. This approach ensures that memory accesses are coalesced and efficient, reducing the overhead caused by scattered memory accesses. By ensuring that data is stored in contiguous memory blocks, CUSNN minimizes memory latency and maximizes GPU utilization.

- **Efficient Memory Management**: The algorithm employs sophisticated memory management strategies to minimize data transfer overhead between the host and device. By keeping the data in the device memory for as long as possible and minimizing host-device communication, CUSNN reduces latency and increases throughput.

- **Scalable Performance**: CUSNN is designed to scale efficiently with larger datasets and higher-dimensional spaces. Thanks to its CUDA optimization and memory management techniques, the algorithm can handle datasets with millions of points and hundreds of dimensions, making it suitable for high-performance computing tasks.

## Installation

### Prerequisites

Before building CUSNN, make sure you have the following tools installed:
- **CUDA Toolkit**: Version 10.0 or above is required to build and run CUSNN.
- **CMake**: To configure and build the project.

### Building CUSNN

To build and run CUSNN, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/nla-group/cusnn.git
   cd cusnn


2. Compile:

```bash
mkdir build
cd build
cmake ..
make
```

3. To link CUSNN with your code, simply: 

```bash
nvcc example.cpp -o example -lcublas -lcusnn -I./include -L./build
```


### License
All the content in this repository is licensed under the MIT License. 


## Reference

Chen X, Güttel S. 2024. Fast and exact fixed-radius neighbor search based on sorting. PeerJ Computer Science 10:e1929 https://doi.org/10.7717/peerj-cs.1929
```
