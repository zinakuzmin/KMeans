#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include "histograma.h"



__global__ void calc2points(double* point_coordinate_1, double* point_coordinate_2 , double* coordinates_arr)
{
    int tid = threadIdx.x; // 52

	coordinates_arr[tid] = (point_coordinate_1[tid] - point_coordinate_2[tid])*(point_coordinate_1[tid] - point_coordinate_2[tid]);
	/*printf("point_coordinate_1[%d]:%.2f\n",tid,point_coordinate_1[tid]);
	printf("point_coordinate_2[%d]:%.2f\n",tid,point_coordinate_2[tid]);*/
	
	//printf("coordinates_arr[%d]:%f\n",tid,coordinates_arr[tid]);

};

//__global__ void calc2pointsWith4Blocks(double* point_coordinate_1, double* point_coordinate_2 , double* coordinates_arr)
//{
//    int tid = threadIdx.x; // 13
//	int bid = blockIdx.x; // 4
//
//	coordinates_arr[tid + 13*bid] = pow(fabs(point_coordinate_1[tid + 13*bid] - point_coordinate_2[tid + 13*bid]),2);
//	/*printf("point_coordinate_1[%d]:%.2f\n",tid,point_coordinate_1[tid]);
//	printf("point_coordinate_2[%d]:%.2f\n",tid,point_coordinate_2[tid]);*/
//	
//	//printf("coordinates_arr[%d]:%f\n",tid,coordinates_arr[tid]);
//
//};



__global__ void calculateDistanceForOneCoordinate(double* points, double* sharedResults , int numOfPoints, int numOfCoordinates)
{
    int col = threadIdx.x; // 52
	int row = blockIdx.x; //811

	
	
	int pos = row*numOfCoordinates+col;


	int block = numOfCoordinates*numOfPoints;

	//Calculate distance between point P1 and all other points (x1-x2)x(x1-x2), (y1-y2)X(y1-y2), ....... and put results in shared temp matrix
	
	for (int i = 0; i < numOfPoints;  i++){

		sharedResults[block*row+i*numOfCoordinates+col] = (points[pos]-points[i*numOfCoordinates+col])*(points[pos]-points[i*numOfCoordinates+col]);
	}
};


__global__ void calculateDistanceBetweenPoints(double* sharedResults , int numOfPoints, int numOfCoordinates, double * results)
{
	//1 block, 811 threads
    int tid = threadIdx.x; 
	//int row = blockIdx.x; 
	int block = numOfCoordinates*numOfPoints;
	int startIdx = tid*block;
	int endIdx = tid*block+block;
	
	double sum = 0;
	//Work on blocks of 811 rows - sum and calc square root
	for (int i = 0; i < numOfPoints; i++){
		for (int j = 0; j < numOfCoordinates; j++){
			sum += sharedResults[startIdx+i*numOfCoordinates+j];
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
        //goto Error;
		error(dev_points, sharedResults, results);
    }




    // Allocate GPU buffers for three vectors (two input, one output)


	// Array of all points
	cudaStatus = cudaMalloc((void**)&dev_points, sizeof(double)*numOfCoordinates*numOfPoints);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
		error(dev_points, sharedResults, results);
    }

	// Results array
	cudaStatus = cudaMalloc((void**)&results, sizeof(double)*numOfPoints*numOfPoints);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
		error(dev_points, sharedResults, results);
    }

	// Shared results array
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

	

    // Launch a kernel on the GPU with one thread for each element.
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


// //Helper function for using CUDA to add vectors in parallel.
//cudaError_t calcDistanceCoordiantesWithCuda(double* coordinates_1, double* coordinates_2, double* coordinates_arr, int num_coordinates, int numOfPoints)
//{
//    double* dev_coordinates_1;
//    double* dev_coordinates_2;
//	double* dev_coordinates_arr;
//	double* dev_points;
//	double* sharedResults;
//	double* results;
//
//	dev_coordinates_1 = 0;//(double*)malloc(sizeof(double)*num_coordinates);
//	dev_coordinates_2 = 0;//(double*)malloc(sizeof(double)*num_coordinates);
//	dev_coordinates_arr = 0;//(double*)malloc(sizeof(double)*num_coordinates);
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        //goto Error;
//		error(dev_coordinates_1, dev_coordinates_2, dev_coordinates_arr);
//    }
//
//
//
//
//    // Allocate GPU buffers for three vectors (two input, one output)
//
//
//	// point_1.coordinates
//	cudaStatus = cudaMalloc((void**)&dev_coordinates_1, sizeof(double)*num_coordinates);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        //goto Error;
//		error(dev_coordinates_1, dev_coordinates_2, dev_coordinates_arr);
//    }
//
//	// point_2.coordinates
//    cudaStatus = cudaMalloc((void**)&dev_coordinates_2, sizeof(double)*num_coordinates);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        //goto Error;
//		error(dev_coordinates_1, dev_coordinates_2, dev_coordinates_arr);
//    }
//
//	// coordinates_arr
//	cudaStatus = cudaMalloc((void**)&dev_coordinates_arr, sizeof(double)*num_coordinates);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        //goto Error;
//		error(dev_coordinates_1, dev_coordinates_2, dev_coordinates_arr);
//    }
//
//
//    // Copy input vectors from host memory to GPU buffers.
//	cudaStatus = cudaMemcpyAsync(dev_coordinates_1, coordinates_1, sizeof(double)*num_coordinates, cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy dev_coordinates_1 failed!");
//        //goto Error;
//		error(dev_coordinates_1, dev_coordinates_2, dev_coordinates_arr);
//    }
//
//	cudaStatus = cudaMemcpyAsync(dev_coordinates_2, coordinates_2, sizeof(double)*num_coordinates, cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy dev_coordinates_2 failed!");
//        //goto Error;
//		error(dev_coordinates_1, dev_coordinates_2, dev_coordinates_arr);
//    }
//
//	cudaStatus = cudaMemcpyAsync(dev_coordinates_arr, coordinates_arr, sizeof(double)*num_coordinates, cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy dev_coordinates_arr failed!");
//		printf("stderr:%s\n",stderr);
//        //goto Error;
//		error(dev_coordinates_1, dev_coordinates_2, dev_coordinates_arr);
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//	calc2points<<<1, 52>>>(dev_coordinates_1, dev_coordinates_2 ,dev_coordinates_arr);
//	//calc2pointsWith4Blocks<<<4, 13>>>(dev_coordinates_1, dev_coordinates_2 ,dev_coordinates_arr);
//    // Check for any errors launching the kernel
//    
//	//cudaStatus = cudaGetLastError();
// //   if (cudaStatus != cudaSuccess) {
// //       fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
// //       //goto Error;
//	//	error(dev_coordinates_1, dev_coordinates_2, dev_coordinates_arr);
// //   }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//  //  cudaStatus = cudaDeviceSynchronize();
//  //  if (cudaStatus != cudaSuccess) {
//  //      fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//  //      //goto Error;
//		//error(dev_coordinates_1, dev_coordinates_2, dev_coordinates_arr);
//  //  }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpyAsync(coordinates_arr, dev_coordinates_arr, sizeof(double)*num_coordinates, cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        //goto Error;
//		error(dev_coordinates_1, dev_coordinates_2, dev_coordinates_arr);
//    }
//	error(dev_coordinates_1, dev_coordinates_2, dev_coordinates_arr);
//	return cudaStatus;
//}




//
//__global__ void initSharedTempResult(double* sharedTempResult, const int arraySize, const int numOfThreads)
//{
//	int threadId = threadIdx.x;
//
//	// Initialize the part of the sharedTempResult that uses the current thread 
//	for(int i = 0; i < arraySize; i++)
//		sharedTempResult[threadId] = 0;
//}
//
//
//
//
//__global__ void calcDistance(double* arr1, double* arr2, double* sharedTempResult, const int numOfThreads, const int arraySize)
//{
//	int threadId = threadIdx.x;	
//
//	//Caclulate distance between each 2 coordinates
//	for(int i = 0; i < numOfThreads; i++)
//		sharedTempResult[threadId] += pow(arr1[i] - arr2[i],2);
//}
//
//void free(double*& dev_arr1, double*& dev_arr2, double*& sharedTempResult)
//{
//	cudaFree(dev_arr1);
//	cudaFree(dev_arr2);
//	cudaFree(sharedTempResult);
//}
//
//
//__global__ void calcSharedDistance(double* sharedTempResult, int arraySize, double* dev_result)
//{
//	int threadId = threadIdx.x;
//	double result = 0;
//	double sum = 0;
//	for (int i = 0; i < arraySize; i++)
//		sum += sharedTempResult[i];
//	result = sqrt(sum);
//	dev_result = &result;
//}
//
//
//cudaError_t CudaCalcDistance( double *arr1,  double *arr2, int arraySize, double * result,  int numOfThreads)
//{
//
//
//
//	double* dev_arr1 = 0;//(double *)malloc(arraySize * sizeof(double));
//    double* dev_arr2 = 0;//(double *)malloc(arraySize * sizeof(double));
//	//double* dev_result = (double *)malloc(sizeof(double));
//	//int* dev_arrSize = (int *)malloc(sizeof(int));
//	//double* sharedTempResult = new double[numOfThreads];
//	double *sharedTempResult = 0 ;//new double[arraySize];
//	//double sharedTempResult[52] = { 0 };
//
//
//
//	//int* dev_numOfThreads = (int *)malloc(sizeof(int));
//	//double* dev_arr1 = (double *)malloc(arraySize * sizeof(double));
// //   double* dev_arr2 = (double *)malloc(arraySize * sizeof(double));
//	//double* dev_result = (double *)malloc(sizeof(double));
//	//int* dev_arrSize = (int *)malloc(sizeof(int));
//	////double* sharedTempResult = new double[numOfThreads];
//	//double *sharedTempResult = (double *)malloc(numOfThreads * sizeof(double));
//	////double sharedTempResult[52] = { 0 };
//    cudaError_t cudaStatus;
//
//	for (int i = 0; i < numOfThreads; i++) {
//		sharedTempResult[i]= 0;
//	}
//
//	for (int i = 0; i < numOfThreads; i++) {
//		fprintf(stderr, "sharedTempResult[%d] = %lf \n" , i, sharedTempResult[i]);
//	}
//
//
//	fprintf(stderr, "\n size of sharedTempResult %d \n", sizeof(sharedTempResult)) ;
//
//	// Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0); // Checking if there is GPU
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//		free(dev_arr1, dev_arr2,sharedTempResult);
//    }
//
//	//// Allocate GPU buffers for arraySize
//	//cudaStatus = cudaMalloc((void**)&dev_arrSize,  sizeof(int));
// //   if (cudaStatus != cudaSuccess) {
// //       fprintf(stderr, "cudaMalloc sharedTempResult failed!");
// //       free(dev_arr1, dev_arr2, sharedTempResult);
// //   }	
//
//	//// Allocate GPU buffers for numOfThreads
//	//cudaStatus = cudaMalloc((void**)&dev_numOfThreads, sizeof(int));
// //   if (cudaStatus != cudaSuccess) {
// //       fprintf(stderr, "cudaMalloc sharedTempResult failed!");
// //       free(dev_arr1, dev_arr2, sharedTempResult);
// //   }	
//
//    // Allocate GPU buffers for array1
//    cudaStatus = cudaMalloc((void**)&dev_arr1, arraySize * sizeof(double));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc arr1 failed!");
//        free(dev_arr1, dev_arr2, sharedTempResult);
//    }
//
//	// Allocate GPU buffers for array2
//    cudaStatus = cudaMalloc((void**)&dev_arr2, arraySize * sizeof(double));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc arr2 failed!");
//        free(dev_arr1, dev_arr2, sharedTempResult);
//    }
//
//	// Allocate GPU buffers for sharedTempResult
//	cudaStatus = cudaMalloc((void**)&sharedTempResult, numOfThreads * sizeof(double));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc sharedTempResult failed!");
//        free(dev_arr1, dev_arr2, sharedTempResult);
//    }	
//
//	//	// Copy input of the numOfThreads from host memory to GPU buffers
// //   cudaStatus = cudaMemcpy(dev_numOfThreads, &numOfThreads, sizeof(int), cudaMemcpyHostToDevice);
// //   if (cudaStatus != cudaSuccess) {
// //       fprintf(stderr, "cudaMemcpy array2 failed!");
// //        free(dev_arr1, dev_arr2, sharedTempResult);
// //   }
//
//
//	//// Copy input of the arraySize from host memory to GPU buffers
// //   cudaStatus = cudaMemcpy(dev_arrSize, &arraySize, sizeof(int), cudaMemcpyHostToDevice);
// //   if (cudaStatus != cudaSuccess) {
// //       fprintf(stderr, "cudaMemcpy array2 failed!");
// //        free(dev_arr1, dev_arr2, sharedTempResult);
// //   }
//
//    // Copy input of the array1 from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_arr1, arr1, arraySize * sizeof(double), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy array1 failed!");
//         free(dev_arr1, dev_arr2, sharedTempResult);
//    }
//
//	// Copy input of the array2 from host memory to GPU buffers
//    cudaStatus = cudaMemcpy(dev_arr2, arr2, arraySize * sizeof(double), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy array2 failed!");
//         free(dev_arr1, dev_arr2, sharedTempResult);
//    }
//
//	
//
//	cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching cudaMemCopy!\n", cudaStatus);
//         free(dev_arr1, dev_arr2, sharedTempResult);
//    }
//
//	// Init the extend histograma
//	initSharedTempResult<<<1, numOfThreads>>>(sharedTempResult, arraySize, numOfThreads);
//
//	// Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "initSharedTempResult launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        free(dev_arr1, dev_arr2, sharedTempResult);
//    }
//
//	cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching initExtendHistograma!\n", cudaStatus);
//        free(dev_arr1, dev_arr2, sharedTempResult);
//    }
//
//    
//	// Calc the histograma
//	calcDistance<<<1, numOfThreads>>>(arr1,arr2,sharedTempResult,numOfThreads, arraySize);
//
//	// Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "calcDistance launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        free(dev_arr1, dev_arr2, sharedTempResult);
//    }
//
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calcDistance!\n", cudaStatus);
//        free(dev_arr1, dev_arr2, sharedTempResult);
//    }
//
//
//
//
//
//	// Calc the extend histograma
//	calcSharedDistance<<<1, 1>>>(sharedTempResult, arraySize, result);
//
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "calcSharedDistance launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        free(dev_arr1, dev_arr2, sharedTempResult);
//    }
//
//	cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calcSharedDistance!\n", cudaStatus);
//        free(dev_arr1, dev_arr2, sharedTempResult);
//    }
//	
//
//
//    //// Copy the result from GPU buffer to host memory.
//    //cudaStatus = cudaMemcpy(result, dev_result, sizeof(double), cudaMemcpyDeviceToHost);
//    //if (cudaStatus != cudaSuccess) {
//    //    fprintf(stderr, "cudaMemcpy calculated result failed!");
//    //    free(dev_arr1, dev_arr2, sharedTempResult);
//    //}
//
//    return cudaStatus;
//}
