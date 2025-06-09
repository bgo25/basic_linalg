// File: matrix_mul_gpu.cu
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>


// For simplicity, let's start with squared matrices
__global__ void matMult(const double* A,
		       const double* B,
		       double* C,
		       int n,
		       int m,
		       int k)
{
	// A: matrix of shape (n, m)
	// B: matrix of shape (m, k)
	// C = A * B, matric of shape (n, k)
	// the operation (i,j) corresponds to computing the element ij
	// of the matrix, obtained by multiplying the i-th row of
	// m1 with the j-th column of m2.
	// So, the total amount of operations are n * m
	// for a matrix with n rows and m columns,
	// (i,j) is identified with "i * m + j" 
	// Basically this is how the operations are indiceded!
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (r < n && c < k) {
		double sum = 0.0;
		// Multiplies the row r of A with the col c of B
		for (int p = 0; p < m; ++p) { 
			// compute C[r, c] = sum_p A_rp * B_pc
			sum += (A[r * m + p] * B[p * k + c]);
		}
		// Store the results into the matrix C
		// C_rc = sum
		C[r * k + c] = sum;
	}
}


void matPrint(const double* M, int n, int m)
{
	std::cout << " --- Matrix of shape (" << n << ", " << m << ") ---";
	std::cout << std::endl;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
      			std::cout << M[i * m + j] << " ";
      		}
		std::cout << std::endl;
	}
}

void matInit(double * M, int n, int m)
{
	for (int i = 0; i < n; ++i) {
		for(int j = 0; j < m; ++j) {
//      			M[i * m + j] = i * m + j;
      			M[i * m + j] = 1.;
     		}
	}
}


int main() {
	// Part0: driving variable
	const int n = 64;
	const int m = 64;
	const int k = 64;

	const size_t bytes_A = n * m * sizeof(double);
	const size_t bytes_B = m * k * sizeof(double);
	const size_t bytes_C = n * k * sizeof(double);

	// Part1: allocating memory on CPU (host)

	std::vector<double> A(n * m);
	std::vector<double> B(m * k);
	std::vector<double> C(n * k);

	matInit(A.data(), n, m);
	matInit(B.data(), m, k);
//	matInit(C.data(), n, k);

	matPrint(A.data(), n, m);
	matPrint(B.data(), m, k);
	matPrint(C.data(), n, k);

	std::cout << "Matrices are initialized on CPU" << std::endl;


	// Part2: allocate memory on GPU (driver)
	double* cuda_A = nullptr;
	double* cuda_B = nullptr;
	double* cuda_C = nullptr;
	cudaMalloc(&cuda_A, bytes_A);
	cudaMalloc(&cuda_B, bytes_B);
	cudaMalloc(&cuda_C, bytes_C);

	// Part3: CPU -> GPU
	std::cout << "CPU -> GPU..." << std::endl;
	cudaMemcpy(cuda_A, A.data(), bytes_A, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_B, B.data(), bytes_B, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_C, C.data(), bytes_C, cudaMemcpyHostToDevice);

	// Part4: launch the Kernel!
	std::cout << "Launching kernel..." << std::endl;
	int dx = k / 32;
	if (k % 32 > 0) {
		dx = dx + 1;
	}
	int dy = n / 16;
	if (dy % 16 > 0) {
		dy += 1;
	}
	dim3 gridSize_3d (dx, dy, 1);
	dim3 blockSize_3d (32, 16, 1);
	matMult<<<gridSize_3d, blockSize_3d>>> (cuda_A,
					 	cuda_B,
					 	cuda_C,
					 	n, m, k);
	//
	
	// Part 5: syncronize and free memory
	cudaDeviceSynchronize();
	std::cout << "GPU -> CPU..." << std::endl;
	cudaMemcpy(C.data(), cuda_C, bytes_C, cudaMemcpyDeviceToHost);
	matPrint(C.data(), n, k);

	cudaFree(cuda_A);
	cudaFree(cuda_B);
	cudaFree(cuda_C);
	std::cout << "END" << std::endl;
	return 0;
}
