#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include "kernel.h"



__global__ void calculateDistanceForOneCoordinate(double* points, double* sharedResults , int numOfPoints, int numOfCoordinates)
{
    int col = threadIdx.x; // 52
	int row = blockIdx.x; //811
	int pos = row*numOfCoordinates+col;
	int block = numOfCoordinates*numOfPoints;

	//Calculate distance between point P1 and all other points (x1-x2)x(x1-x2), (y1-y2)X(y1-y2), ....... and put results in shared temp matrix
	
	for (int i = 0; i < numOfPoints;  i++){

		sharedResults[(block*row)+(i*numOfCoordinates)+col] = (points[pos]-points[i*numOfCoordinates+col])*(points[pos]-points[i*numOfCoordinates+col]);
	}
};


__global__ void calculateDistanceBetweenPoints(double* sharedResults , int numOfPoints, int numOfCoordinates, double * results)
{
	//1 block, 811 threads
    int tid = threadIdx.x;  
	int block = numOfCoordinates*numOfPoints;
	int startIdx = tid*block;
	//int endIdx = tid*block+block;
	
	double sum = 0;
	//Work on blocks of 811 rows - sum and calc square root
	for (int i = 0; i < numOfPoints; i++){
		sum = 0;
		for (int j = 0; j < numOfCoordinates; j++){
			sum += sharedResults[startIdx+(i*numOfCoordinates)+j];
		}

		results[tid*numOfPoints+i] = sqrt(sum);
	}

};


void error(double* coordinate_1 ,double* coordinate_2, double* coordinates_arr)
{
	cudaFree(coordinate_1);
	cudaFree(coordinate_2);
	cudaFree(coordinates_arr);
	
}



 //Helper function for using CUDA to add vectors in parallel.
cudaError_t calcDistanceWithCuda(double* points, double* resultsFromCuda, int numOfCoordinates, int numOfPoints)
{
 
	double* dev_points;
	double* sharedResults;
	double* results;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		//error(dev_points, sharedResults, results);
    }



	// Allocate GPU buffers for Array of all points
	cudaStatus = cudaMalloc((void**)&dev_points, sizeof(double)*numOfCoordinates*numOfPoints);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
		error(dev_points, sharedResults, results);
    }

	// Allocate GPU buffers for Results array
	cudaStatus = cudaMalloc((void**)&results, sizeof(double)*numOfPoints*numOfPoints);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
		error(dev_points, sharedResults, results);
    }

	// Allocate GPU buffers for Shared results array
	cudaStatus = cudaMalloc((void**)&sharedResults, sizeof(double)*numOfCoordinates*numOfPoints*numOfPoints);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
		error(dev_points, sharedResults, results);
    }


    // Copy input array of points from host memory to GPU buffers.
	cudaStatus = cudaMemcpyAsync(dev_points, points, sizeof(double)*numOfCoordinates*numOfPoints, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy dev_coordinates_1 failed!");
		error(dev_points, sharedResults, results);
    }

	

    // Launch a kernel on the GPU to calculate partial distances by coordinate .
	calculateDistanceForOneCoordinate<<<numOfPoints, numOfCoordinates>>>(dev_points, sharedResults, numOfPoints, numOfCoordinates);

	
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "calcExtendHistograma launch failed: %s\n", cudaGetErrorString(cudaStatus));
        error(dev_points, sharedResults, results);
    }

	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calcExtendHistograma!\n", cudaStatus);
       error(dev_points, sharedResults, results);
    }

	// Launch a kernel on the GPU to calculate distances between each 2 points
	calculateDistanceBetweenPoints<<<1, numOfPoints>>>(sharedResults, numOfPoints, numOfCoordinates, results);


   // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "calcExtendHistograma launch failed: %s\n", cudaGetErrorString(cudaStatus));
        error(dev_points, sharedResults, results);
    }

	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calcExtendHistograma!\n", cudaStatus);
       error(dev_points, sharedResults, results);
    }


    // Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpyAsync(resultsFromCuda, results, sizeof(double)*numOfPoints*numOfPoints, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        //goto Error;
		error(dev_points, sharedResults, results);
    }
	
	return cudaStatus;
}
