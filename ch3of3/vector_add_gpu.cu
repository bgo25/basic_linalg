// Writing a first example using CUDA.
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

/* This function sums only a specific index in the vector, and the
* final actual sum will be the results of the parallelization */

__global__ void addVectors(const double* a,
			   const double* b,
			   double* c,
			   int n)
{
	// Single identifier determined when the thread is lauched
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	// It is possible that *more* threads are produced,
	// so we add a check to ensure that the current will work
	// on legit indeces
	if (index < n) {
		c[index] = a[index] + b[index];
	}
}

int main() {
	// Part1: allocate memory on the Host (CPU)
	const int n = 100000;
	const size_t bytes = n * sizeof(double);
	// Simply declaring and initializating three vectors
	std::vector<double> h_a(n);
	std::vector<double> h_b(n);
	std::vector<double> h_c(n);
	for (int i = 0; i < n; ++i) {
		h_a[i] = i;
		h_b[i] = i * 2.0;
	}

	// Part2: allocate memory on device (GPU)
	double* d_a = nullptr;
	double* d_b = nullptr;
	double* d_c = nullptr;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	// Part3: tranfer data FROM host TO device
	cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

	// Part 4: launch the KERNEL
	int blockSize = 256; // Common value for number of threads per block
	int gridSize = int(n / blockSize) + blockSize;
	std::cout << "Launching CUDA: [gridSize " << gridSize;
	std::cout << ", blockSize " << blockSize << "]" << std::endl;
	addVectors<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err);
		std::cerr << std::endl;
	}


	// Part 5: Transfer the reults to CPU
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err);
		std::cerr << std::endl;
	}

	// For simplicity, I am NOT managing the errors

	std::cout << "Transfering GPU -> CPU " << std::endl;
	cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

	bool flag = true;
	for (int i = 0; i < n; ++i) {
		if (std::abs(h_a[i] + h_b[i] - h_c[i]) > 1e-9) {
      			std::cout << "ERR! Index" << i << " :";
      			std::cout << h_a[i] + h_b[i] << "!=" << h_c[i];
			std::cout << std::endl;
      			flag = false;
      		}
	}
	if (flag) {
		std::cout << "SUCCESS" << std::endl;
	} else {
		std::cout << "FAIL" << std::endl;
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
