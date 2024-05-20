
#include "fir_gpu.h"
#include "cuda_timer.h"

#include <iostream>

#define BLOCK_SIZE 64
#define NUM_BLOCKS 128


// Baseline
__global__
void fir_kernel1(const float *coeffs, const float *input, float *output, int length, int filterLength)
{
	int tx = threadIdx.x;
    int bx = blockIdx.x;
    
    int bs = blockDim.x;
    int gs = gridDim.x;

    int tid = bs*bx + tx;

	// Apply the filter to each input sample
	for (int n = bx; n < length-filterLength; n = n + gs )
	{
		// Calculate output n
		float acc = 0;
		for (int k = tx; k < filterLength; k = k + bs)
		{
			acc += coeffs[k] * input[n + k];
		}
		output[n] = acc;
	}

}


// Coefficients in shared memory
// Here we suppose that filterLength and BLOCK_SIZE is always 64
__global__
void fir_kernel2(const float *coeffs, const float *input, float *output, int length, int filterLength)
{
	__shared__ float sm_coeffs[64]; //static allocation of shared memory

	int tx = threadIdx.x;
    int bx = blockIdx.x;
    
    int bs = blockDim.x;
    int gs = gridDim.x;

	int tid = bs*bx + tx;
	int threads = bs*gs;

	for (int k = tid; k < filterLength; k = k + threads){
		sm_coeffs[k] = coeffs[k];
	}

	for (int n = bx; n < length-filterLength; n = n + gs )
	{
		// Calculate output n
		float acc = 0;
		for (int k = tx; k < filterLength; k = k + bs)
		{
			acc += sm_coeffs[k] * input[n + k];
		}
		output[n] = acc;
	}


	
}


// Coefficients and inputs in shared memory
// Here we suppose that filterLength and BLOCK_SIZE is always 64
__global__
void fir_kernel3(const float *coeffs, const float *input, float *output, int length, int filterLength)
{
	__shared__ float sm_coeffs[64]; //static allocation of shared memory
	__shared__ float sm_inputs[64]; //static allocation of shared memory

	int tx = threadIdx.x;
    int bx = blockIdx.x;
    
    int bs = blockDim.x;
    int gs = gridDim.x;

	int tid = bs*bx + tx;
	int threads = bs*gs;

	for (int k = tid; k < filterLength; k = k + threads){
		sm_coeffs[k] = coeffs[k];
	}

	for (int k = tid; k < length; k = k + threads){
		sm_inputs[k] = input[k];
	}

	for (int n = bx; n < length-filterLength; n = n + gs )
	{
		// Calculate output n
		float acc = 0;
		for (int k = tx; k < filterLength; k = k + bs)
		{
			acc += sm_coeffs[k] * sm_inputs[n + k];
		}
		output[n] = acc;
	}

	
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
	const int num_blocks = NUM_BLOCKS;

    dim3 block(block_size, 1, 1);
    dim3 grid(num_blocks, 1, 1);

	timer.start();
	// fir_kernel1<<<num_blocks, block_size>>>(coeffs, input, output, length, filterLength);
	// fir_kernel2<<<num_blocks, block_size>>>(coeffs, input, output, length, filterLength);
	fir_kernel3<<<num_blocks, block_size>>>(coeffs, input, output, length, filterLength);
	timer.stop();

	cudaDeviceSynchronize();

	CudaCheckError();

	// float time_gpu_kernel1 = timer.getElapsed();
		
	// timer.start();
	// fir_kernel2<<<num_blocks, block_size>>>(coeffs, input, output, length, filterLength);
	// timer.stop();

	// cudaDeviceSynchronize();

	// CudaCheckError();
	// float time_gpu_kernel2 = timer.getElapsed();


	// timer.start();
	// fir_kernel3<<<num_blocks, block_size>>>(coeffs, input, output, length, filterLength);
	// timer.stop();

	// cudaDeviceSynchronize();

	// CudaCheckError();
	// float time_gpu_kernel3 = timer.getElapsed();

	//std::cout << "Kernel Time: " << time_gpu << "ms\n";
}



