#include "cuda_runtime.h"

// Consts
const int NUMBERS_ARR_SIZE = 100000;
const int RANGE_OF_NUMBERS = 256;
const int MPI_PROCS = 2;
const int OMP_THREADS = 2;
const int CUDE_THREADS = 25;
const int SHARED_HISTOGRAMA_SIZE = RANGE_OF_NUMBERS * MPI_PROCS;


//cudaError_t CudaCalcDistance(  double *arr1,  double *arr2,  int arraySize, double * result,  int numOfThreads);
cudaError_t calcDistanceCoordiantesWithCuda(double* coordinates_1, double* coordinates_2, double* coordinates_arr, int num_coordinates);
cudaError_t calcDistanceWithCuda(double* points, double* resultsFromCuda, int numOfCoordinates, int numOfPoints);