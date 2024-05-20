
#include "fir_gpu.h"
#include "cuda_timer.h"

#include <iostream>

#define BLOCK_SIZE 64
// #define NUM_BLOCKS 128

extern __shared__ float smem[];

// Baseline
__global__
void fir_kernel1(const float *coeffs, const float *input, float *output, int length, int filterLength)
{
	int tx = threadIdx.x;
    int bx = blockIdx.x;
    
    int bs = blockDim.x;
    int gs = gridDim.x;

    int tid = bs*bx + tx;
	int threads = bs*gs;

	// Apply the filter to each input sample
	// for (int n = bx; n < length-filterLength; n = n + gs )
	for (int n = tid; n < length-filterLength; n = n + threads )
	{
		__syncthreads();
		// Calculate output n
		float acc = 0;
		// for (int k = tx; k < filterLength; k = k + bs)
		for (int k = 0; k < filterLength; k++)

		{
			acc += coeffs[k] * input[n + k];
		}

		__syncthreads();
		output[n] = acc;
	}

}


// Coefficients in shared memory
// Here we suppose that filterLength and BLOCK_SIZE is always 64
__global__
void fir_kernel2(const float *coeffs, const float *input, float *output, int length, int filterLength)
{
	// __shared__ float sm_coeffs[]; //static allocation of shared memory

	float* sm_coeffs = &smem[0];
	int tx = threadIdx.x;
    int bx = blockIdx.x;
    
    int bs = blockDim.x;
    int gs = gridDim.x;

	int tid = bs*bx + tx;
	int threads = bs*gs;

	// for (int k = tx; k < filterLength; k = k + bs){
		sm_coeffs[tx] = coeffs[tx];
	// }

	__syncthreads();

	// for (int n = bx; n < length-filterLength; n = n + gs )
	// for (int n = tid; n < length-filterLength; n = n + threads )

	// {
		// Calculate output n
		float acc = 0;
		// for (int k = tx; k < filterLength; k = k + bs)
		for (int k = 0; k < filterLength; k++)

		{
			acc += sm_coeffs[k] * input[tid + k];
		}
		__syncthreads();

		output[tid] = acc;

	// }


	
}


// Coefficients and inputs in shared memory
// Here we suppose that filterLength and BLOCK_SIZE is always 64
__global__
void fir_kernel3(const float *coeffs, const float *input, float *output, int length, int filterLength)
{
	float* sm_coeffs = &smem[0];
	float* sm_inputs = &smem[filterLength];

	int tx = threadIdx.x;
    int bx = blockIdx.x;
    
    int bs = blockDim.x;
    int gs = gridDim.x;

	int tid = bs*bx + tx;
	int threads = bs*gs;

	// for (int k = tx; k < filterLength; k = k + bs){
		sm_coeffs[tx] = coeffs[tx];
	// }

	// for (int k = tx; k <= length/gs; k = k + bs){
		sm_inputs[tx] = input[bx*bs + tx];
		sm_inputs[tx+bs] = input[(bx+1)*bs + tx];
	// }

	__syncthreads();

	// input += tx + BLOCK_SIZE;
	// output += tx + BLOCK_SIZE;

	// for (int n = bx; n < length-filterLength; n = n + gs ){
	// for (int n = tx; n < length-filterLength; n = n + bs ){
		// Calculate output n
		float acc = 0;
		for (int k = 0; k < filterLength; k++)
		// for (int k = 0; k < filterLength; k++)
		{
			acc += sm_coeffs[k] * sm_inputs[tx + k];
		}

		__syncthreads();
		// output[n + blockIdx.x * bs] = acc;
		output[bx*bs + tx] = acc;
	// }

	
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
	int num_blocks = length/block_size;

	if(length % block_size){
		num_blocks++;
	}

	printf("input length: %d, num_blocks: %d\n", length, num_blocks);

    dim3 block(block_size, 1, 1);
    dim3 grid(num_blocks, 1, 1);

	timer.start();
	fir_kernel1<<<num_blocks, block_size>>>(coeffs, input, output, length, filterLength);
	// fir_kernel2<<<num_blocks, block_size, 1024*48>>>(coeffs, input, output, length, filterLength);
	// fir_kernel3<<<num_blocks, block_size, 1024*48>>>(coeffs, input, output, length, filterLength);
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



