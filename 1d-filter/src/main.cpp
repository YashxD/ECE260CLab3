#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include <cmath>
#include <fstream>
#include <vector>

#include "main.h"
#include "fir_gpu.h"
#include "timer.h"

using namespace std;


struct Profile
{
	float time_init;
	float time_cpu;
	float time_gpu;
	float time_rmse;
	float time_total;

	Profile():time_init(0), time_cpu(0), time_gpu(0), time_rmse(0), time_total(0){}
};


Profile computeProfileAverage(const vector<Profile>& p, int n)
{
	n = min((int)p.size(), n);

	Profile avg;
	for(int i = p.size()-1; i >= (int)p.size() - n; i--)
	{
		avg.time_init += p[i].time_init;
		avg.time_cpu += p[i].time_cpu;
		avg.time_gpu += p[i].time_gpu;
		avg.time_rmse += p[i].time_rmse;
	}
	avg.time_init /= n; 
	avg.time_cpu  /= n; 
	avg.time_gpu  /= n; 
	avg.time_rmse /= n; 
	avg.time_total = avg.time_init + avg.time_cpu + avg.time_gpu + avg.time_rmse;

	return avg;
}


// ----------------------------------------------
// Run a FIR filter on the given input data
// ----------------------------------------------
void fir_cpu(const float *coeffs, const float *input, float *output, int length, int filterLength)
// ----------------------------------------------
{
	// Apply the filter to each input sample
	for (int n = 0; n < length-filterLength; n++)
	{
		// Calculate output n
		float acc = 0;
		for (int k = 0; k < filterLength; k++)
		{
			acc += coeffs[k] * input[n + k];
		}
		output[n] = acc;
	}
}

// ----------------------------------------------
// Create filter coefficients
// ----------------------------------------------
void designLPF(float* coeffs, int filterLength, float Fs, float Fx)
// ----------------------------------------------
{
	float lambda = M_PI * Fx / (Fs/2);

	for(int n = 0; n < filterLength; n++)
	{
		float mm = n - (filterLength - 1.0) / 2.0;
		if( mm == 0.0 ) coeffs[n] = lambda / M_PI;
		else coeffs[n] = sin( mm * lambda ) / (mm * M_PI);
	}
}


// ----------------------------------------------
int main( void )
// ----------------------------------------------
{
	const int SAMPLES    = 100000; // Size of input data
	const int FILTER_LEN = 64;     // Size of filter
	const int PADDED_SIZE = SAMPLES + 2*FILTER_LEN; // Size of padded input
	const int OUTPUT_SIZE = SAMPLES + FILTER_LEN;   // Size of output data
	cudaError_t err;
	
	vector<Profile> profiler;


	CudaSafeCall(cudaSetDeviceFlags(cudaDeviceMapHost));


	float input[SAMPLES];
	float output1[OUTPUT_SIZE];

	float* paddedInput;
	float* coeffs;
	float* output2;

	float* d_paddedInput;
	float* d_coeffs;
	float* d_output2;

	// TODO Replace by CUDA unified memory
	coeffs      = (float*)malloc(FILTER_LEN * sizeof(float));
	paddedInput = (float*)malloc(PADDED_SIZE * sizeof(float));
	output2     = (float*)malloc(OUTPUT_SIZE * sizeof(float));

	err = cudaMalloc(&d_coeffs, FILTER_LEN * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

	err = cudaMalloc(&d_paddedInput, PADDED_SIZE * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

	err = cudaMalloc(&d_output2, OUTPUT_SIZE * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }


	// Initialize coefficients
	designLPF(coeffs, FILTER_LEN, 44.1, 2.0);

    err = cudaMemcpy(d_coeffs, coeffs, FILTER_LEN * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR 143: cudaMemCpy failed!\n");
        exit(1);
    }

	LinuxTimer timer;


	for(int nruns = 0; nruns < 100; nruns++)
	{
		profiler.push_back(Profile());


		timer.start();

		// Initialize inputs
		for(int i = 0; i < SAMPLES; i++)
		{
			input[i] = rand() / (float)RAND_MAX;
		}


		// Pad inputs
		for(int i = 0; i < PADDED_SIZE; i++)
		{
			if(i < FILTER_LEN || i >= SAMPLES+FILTER_LEN)
			{
				paddedInput[i] = 0;
			}
			else{ paddedInput[i] = input[i - FILTER_LEN]; }
		}

		timer.stop();

		profiler.back().time_init = timer.getElapsed() / 1000000.f;

		err = cudaMemcpy(d_paddedInput, paddedInput, PADDED_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    	if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR 180: cudaMemCpy failed!\n");
        exit(1);
    	}

		// cudaFuncSetAttribute(fir_kernel2, cudaFuncAttributePreferredSharedMemoryCarveout,cudaSharedmemCarveoutMaxShared);
    	// checkCUDAError("Error seting shared memory to 64K carveout");

		// FIR on CPU
		timer.start();
		fir_cpu(coeffs, paddedInput, output1, PADDED_SIZE, FILTER_LEN);
		timer.stop();

		profiler.back().time_cpu = timer.getElapsed() / 1000000.f;


		// FIR on GPU
		timer.start();
		fir_gpu(d_coeffs, d_paddedInput, d_output2, PADDED_SIZE, FILTER_LEN);
		timer.stop();

		profiler.back().time_gpu = timer.getElapsed() / 1000000.f;

		//Copy output from device to host
		// err = cudaMemcpy(d_output2, output2, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    	// if (err != cudaSuccess) {
        // fprintf(stderr, "GPU_ERROR 202: cudaMemCpy failed!\n");
        // exit(1);
    	// }

		timer.start();
		
		// Check for errors
		float mse = 0.f;
		for(int i = 0; i < OUTPUT_SIZE; i++)
		{
			float diff = output1[i] - output2[i];
			mse += diff * diff;
		}
		mse /= OUTPUT_SIZE;

		timer.stop();

		profiler.back().time_rmse = timer.getElapsed() / 1000000.f;


		Profile avg = computeProfileAverage(profiler, 20);
		printf("Iteration: %d, Time Init: %.2fms  Time CPU: %.2fms  Time GPU: %.2fms  Time RMSE: %.2fms  Time total: %.2fms  RMSE: %.3f\n",
				nruns, avg.time_init, avg.time_cpu, avg.time_gpu, avg.time_rmse, avg.time_total, sqrt(mse));
		

		// Print result
		//printArray(input, SAMPLES, false);
		//cout << "\n";
		//printArray(paddedInput, PADDED_SIZE, false);
		//cout << "\n";
		//printArray(output1, OUTPUT_SIZE, false);
		//cout << "\n";
		//printArray(output2, OUTPUT_SIZE, false);
		//cout << "\n";
		//cout << endl;
	}

	// TODO free with cuda
	cudaFree(d_paddedInput);
	cudaFree(d_output2);
	cudaFree(d_coeffs);
	free(paddedInput);
	free(output2);
	free(coeffs);

	return 0;
}



