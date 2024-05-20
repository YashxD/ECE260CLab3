#ifndef __FIR_GPU_H__
#define __FIR_GPU_H__


#include <cuda_runtime.h>

#include "cuda_error.h"


void fir_gpu(const float *coeffs, const float *input, float *output, int length, int filterLength);


#endif // __FIR_GPU_H__
