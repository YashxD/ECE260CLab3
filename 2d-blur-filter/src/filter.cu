#include "filter.h"
#include "cuda_timer.h"

#include <iostream>

using namespace std;

#define FILTER_SIZE 3
#define FILTER_LIM FILTER_SIZE/2

__global__
void kernel_filter(const uchar * input, uchar * output, const uint height, const uint width)
{
	// TODO: Implement a blur filter for the camera (averaging an NxN array of pixels

  // uint8_t filterSize = 3;
  	int tx = threadIdx.x;
    int bx = blockIdx.x;
    
    int bs = blockDim.x;
    int gs = gridDim.x;

    int tid = bs*bx + tx;
	int threads = bs*gs;
  uint16_t kernelSum = 0;

  for (int i = FILTER_LIM + bx; i < height + FILTER_LIM; i = i + gs) {
    for (int j = FILTER_LIM + tx; j < width + FILTER_LIM; j = j + bs) {
      kernelSum = 0;
      // Add all the elements to get the sum
	  #pragma unroll
      for (int ii = i - FILTER_LIM; ii < i + FILTER_LIM; ii++) {
		#pragma unroll
        for (int jj = j - FILTER_LIM; jj < j + FILTER_LIM; jj++) {
          kernelSum += input[(ii * width) + jj];
        }
      }
      output[(i * width) + j] = kernelSum/(FILTER_SIZE * FILTER_SIZE);
    }
  }
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

	dim3 grid(64, 1, 1);  // TODO
	dim3 block(256, 1, 1); // TODO

	timer.start();
	kernel_filter<<<grid, block>>>(input, output, height, width);
	timer.stop();

	cudaDeviceSynchronize();

	float time_kernel = timer.getElapsed();
}





