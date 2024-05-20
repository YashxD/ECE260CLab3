
#include "fir_gpu.h"
#include "cuda_timer.h"

#include <iostream>

#define BLOCK_SIZE 64


// Baseline
__global__
void fir_kernel1(const float *coeffs, const float *input, float *output, int length, int filterLength)
{
	// TODO
}


// Coefficients in shared memory
// Here we suppose that filterLength and BLOCK_SIZE is always 64
__global__
void fir_kernel2(const float *coeffs, const float *input, float *output, int length, int filterLength)
{
	// TODO
}


// Coefficients and inputs in shared memory
// Here we suppose that filterLength and BLOCK_SIZE is always 64
__global__
void fir_kernel3(const float *coeffs, const float *input, float *output, int length, int filterLength)
{
	// TODO
}



inline int divup(int a, int b)
{
	if (a % b)
		return a / b + 1;
	else
		return a / b;
}

void fir_gpu(const float *coeffs, const float *input, float *output, int length, int filterLength)
{
	const int output_size = length - filterLength;

	CudaSynchronizedTimer timer;

	const int block_size = BLOCK_SIZE;

    dim3 block(block_size, 1, 1);
    dim3 grid(/*TODO*/1, 1, 1);

	timer.start();
	// TODO Launch kernel here
	timer.stop();

	cudaDeviceSynchronize();

	CudaCheckError();

	float time_gpu = timer.getElapsed();
	
	//std::cout << "Kernel Time: " << time_gpu << "ms\n";
}



