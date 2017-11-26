#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include "histograma.h"



__global__ void initSharedTempResult(double* sharedTempResult, const int arraySize, const int numOfThreads)
{
	int threadId = threadIdx.x;

	// Initialize the part of the sharedTempResult that uses the current thread 
	for(int i = 0; i < arraySize; i++)
		sharedTempResult[threadId] = 0;
}




__global__ void calcDistance(double* arr1, double* arr2, double* sharedTempResult, const int numOfThreads, const int arraySize)
{
	int threadId = threadIdx.x;	

	// Summarize the shared histograma into the final histograma
	for(int i = 0; i < numOfThreads; i++)
		sharedTempResult[threadId] += pow(arr1[i] - arr2[i],2);
}

void free(double*& dev_arr1, double*& dev_arr2, double*& sharedTempResult)
{
	cudaFree(dev_arr1);
	cudaFree(dev_arr2);
	cudaFree(sharedTempResult);
}


__global__ void calcSharedDistance(double* sharedTempResult, int arraySize, double* dev_result)
{
	int threadId = threadIdx.x;
	double result = 0;
	double sum = 0;
	for (int i = 0; i < arraySize; i++)
		sum += sharedTempResult[i];
	result = sqrt(sum);
	dev_result = &result;
}


cudaError_t CudaCalcDistance( double *arr1,  double *arr2, int arraySize, double * result,  int numOfThreads)
{
	double* dev_arr1 = 0;
    double* dev_arr2 = 0;
	double* dev_result = 0;
	double* sharedTempResult = new double[numOfThreads];
    cudaError_t cudaStatus;


	// Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0); // Checking if there is GPU
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		free(dev_arr1, dev_arr2,sharedTempResult);
    }

    // Allocate GPU buffers for array1
    cudaStatus = cudaMalloc((void**)&dev_arr1, arraySize * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc array failed!");
        free(dev_arr1, dev_arr2, sharedTempResult);
    }

	// Allocate GPU buffers for array2
    cudaStatus = cudaMalloc((void**)&dev_arr1, arraySize * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc histograma failed!");
        free(dev_arr1, dev_arr2, sharedTempResult);
    }

	// Allocate GPU buffers for sharedTempResult
	cudaStatus = cudaMalloc((void**)&sharedTempResult, numOfThreads * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc extend histograma failed!");
        free(dev_arr1, dev_arr2, sharedTempResult);
    }	

    // Copy input of the array from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_arr1, arr1, arraySize * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy array failed!");
         free(dev_arr1, dev_arr2, sharedTempResult);
    }

	// Copy input of the histograma from host memory to GPU buffers
    cudaStatus = cudaMemcpy(dev_arr2, arr2, arraySize * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy histograma failed!");
         free(dev_arr1, dev_arr2, sharedTempResult);
    }

	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching initExtendHistograma!\n", cudaStatus);
         free(dev_arr1, dev_arr2, sharedTempResult);
    }

	// Init the extend histograma
	initSharedTempResult<<<1, numOfThreads>>>(sharedTempResult, arraySize, numOfThreads);

	// Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "initExtendHistograma launch failed: %s\n", cudaGetErrorString(cudaStatus));
        free(dev_arr1, dev_arr2, sharedTempResult);
    }

	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching initExtendHistograma!\n", cudaStatus);
        free(dev_arr1, dev_arr2, sharedTempResult);
    }

    // Calc the extend histograma
	calcSharedDistance<<<1, numOfThreads>>>(sharedTempResult, arraySize, dev_result);


    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "calcExtendHistograma launch failed: %s\n", cudaGetErrorString(cudaStatus));
        free(dev_arr1, dev_arr2, sharedTempResult);
    }

	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calcExtendHistograma!\n", cudaStatus);
        free(dev_arr1, dev_arr2, sharedTempResult);
    }
	
	// Calc the histograma
	calcDistance<<<1, 1>>>(arr1,arr2,sharedTempResult,numOfThreads, arraySize);

	// Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "calcHistograma launch failed: %s\n", cudaGetErrorString(cudaStatus));
        free(dev_arr1, dev_arr2, sharedTempResult);
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calcHistograma!\n", cudaStatus);
        free(dev_arr1, dev_arr2, sharedTempResult);
    }

    // Copy the result from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(result, dev_result, sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy calculated histograma failed!");
        free(dev_arr1, dev_arr2, sharedTempResult);
    }

    return cudaStatus;
}
