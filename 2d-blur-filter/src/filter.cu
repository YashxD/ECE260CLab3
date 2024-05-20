#include "filter.h"
#include "cuda_timer.h"

#include <iostream>

using namespace std;

__global__
void kernel_filter(const uchar * input, uchar * output, const uint height, const uint width)
{
	// TODO: Implement a blur filter for the camera (averaging an NxN array of pixels
}

inline int divup(int a, int b)
{
	if (a % b)
		return a / b + 1;
	else
		return a / b;
}

/**
 * Wrapper for calling the kernel.
 */
void filter_gpu(const uchar * input, uchar * output, const uint height, const uint width)
{
	CudaSynchronizedTimer timer;

	// Launch the kernel
	const int grid_x = 64;
	const int grid_y = 64;

	dim3 grid(1, 1, 1);  // TODO
	dim3 block(1, 1, 1); // TODO

	timer.start();
	kernel_filter<<<grid, block>>>(input, output, height, width);
	timer.stop();

	cudaDeviceSynchronize();

	float time_kernel = timer.getElapsed();
}





