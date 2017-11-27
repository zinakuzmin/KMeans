// 327366746 Zina Kuzmin
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include <math.h>
#include "FileHandler.h"
#include <ctime>
#include <concrt.h>

#include "histograma.h"

//#pragma threadprivate(tid)

void accociatePointsToCluster(vector<Point> cluster_center, vector<Point> *points);
double calculateDistanceBetweenPoints(Point point1, Point point2);
double calculateClusterDiameter(vector<Point> points, vector<Point> cluster_centers, int clusterIdx);
double calculateQualityMeature(vector<Point> points, vector<Point> clusterCenters);
void printPoints(vector<Point> points);
void countPointsInCluster(int numOfCenters, vector<Point> points);
int reaccociatePointsToCluster(vector<Point> cluster_center, vector<Point>* points);
void calculateClusterCenter(vector<Point> * cluster_centers, vector<Point> points);
void findAllPointsInCluster(vector<Point> points, vector<Point> * pointsInCluster, int clusterIdx);
double* makeArray(double * arr, vector<double> coordinates);

/*k means
1. Start MPI
2. As MPI master read once first line of input file to get the parameters
3. As MPI master read once all products till end of file and put all data to vector
4. As MPI master take 2 first rows of matrix and set them centers
5. Send centers to MPI slaves
6. Send parts if the matrix to MPI slaves
7. Each MPI will associate his part of products to the relevant center

*/



int main(int argc,char *argv[])
{	
	int numprocs;
	int myid;
	/*MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);	
	MPI_Status status;*/


	int N = 0;   //Number of products
	int n = 0;   //Dimention
	int MAX = 0;  //Maximum number of clusters to find
	int LIMIT = 0; //Maximum number of iterations of kmeans run
	double QM = 0;  //Quality measure
	vector<Point> points;
	vector<Point> cluster_centers;
	char filepath[] = ("dataset_normalized.txt");
	//char filepath[] = ("testDataSet.txt");
	int number_of_clusters = 2;
	omp_set_num_threads(8);


	//check that MPI is running this program only with 3 threads:
	/*if(numprocs != MPI_PROCS)
	{
	printf("Please run with 3 threads.\n");
	MPI_Abort(MPI_COMM_WORLD, 1);
	}*/

	fflush(stdout);

	//MPI master
	//if (myid == 0) 
	{
		//Read data from input file which is located in project folder
	//	points.resize(1000);
		readFromFile(filepath, &N, &n, &MAX, &LIMIT, &QM, &points);


		

		/*for (int i = 0; i < number_of_clusters ; i++)
		{
			cout << "Original Center " << i << ". Coordinates: " << cluster_centers[i].coordinates[0] << ", " << cluster_centers[i].coordinates[1] << ".\n";
		}*/

		//cout << "Center coordinates "<< endl;
		//printPoints(cluster_centers);


		//accociatePointsToCluster(cluster_centers, &points);
		//cout << "Points after associating" << endl;
		//printPoints(points);
		//cout << "QM: " << calculateQualityMeature(points, cluster_centers) <<endl;


		//time_t begin,end; // time_t is a datatype to store time values.

		//time (&begin); // note time before execution
		
		const clock_t begin_time = clock();
		
		


		double qm = 0;

		do{
			cluster_centers.clear();
			cout << "****************************************" << endl;
			cout << "Round k=" << number_of_clusters << endl;
			//Choose first K points as clusters centers
			for (int i = 0;i < number_of_clusters;i++){
			cluster_centers.push_back (points[i]);
			}

			cout << "*******************Print initial centers*********************" << endl;
			printPoints(cluster_centers);

			accociatePointsToCluster(cluster_centers, &points);
			countPointsInCluster(number_of_clusters, points);

			int pointMoved = 0;
			int iteration = 0;
			do{
				iteration++;
				calculateClusterCenter(&cluster_centers, points);
				cout << "*******************Print centers after recalc*********************" << endl;
				printPoints(cluster_centers);
				pointMoved = reaccociatePointsToCluster(cluster_centers, &points);
				cout << "After recalc points were moved " << pointMoved << endl;
				countPointsInCluster(number_of_clusters, points);

			}
			while (pointMoved != 0 && iteration < LIMIT);

			qm = calculateQualityMeature(points, cluster_centers);
			cout << "calculated quality is " << qm << " and expected " << QM << endl; 
			number_of_clusters++;
		}
		while (QM < qm && number_of_clusters <= MAX);


		
		//time (&end); // note time after execution

		//double difference = difftime (end,begin);
		//printf ("time taken  %.2lf seconds.\n", difference );
		std::cout << "execution time " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;


		//write results to file in project folder
		//writeResultsToFile("results.txt", points, QM, MAX);

		cout<<"Execution completed, press return\n";


	}

	/*else {


	}*/

	//MPI_Finalize();
	return 0;
}


void printPoints(vector<Point> points){
	clock_t begin = clock();
	cout << "##############################" << endl;
	for (int i = 0 ; i < points.size(); i++){
		points[i].toString();
	}

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	cout << "Time to execute printPoints "<< elapsed_secs << endl;

}



void countPointsInCluster(int numOfCenters, vector<Point> points){
	clock_t begin = clock();
	int count = 0;

	for (int j = 0; j < numOfCenters; j++){
		for (int i = 0; i < points.size(); i++){
			if (points[i].center_id == j)
				count++;
		}

		cout << "Center " << j << " has " << count << "points" << endl;
		count = 0;

	}
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	cout << "Time to execute countPointsInCluster "<< elapsed_secs << endl;
}


//Set center field of point to the closest cluster center
void accociatePointsToCluster(vector<Point> cluster_center, vector<Point>* points){
	clock_t begin = clock();
	double minDistance = 0;
	double distance = 0;
	int clusterIdx = -1;

	//#pragma omp parallel for 
	for (vector<Point>::iterator it = points->begin() ; it != points->end(); ++it){
		for (int i = 0 ; i < cluster_center.size(); i++){
			distance = calculateDistanceBetweenPoints(cluster_center[i], *it);
			
			if (i==0){
				minDistance = distance;
				clusterIdx = i;
			}

			if (minDistance > distance){
				minDistance = distance;
				clusterIdx = i;
			}
		}
		(*it).center_id = clusterIdx;
		

	}

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	cout << "Time to execute accociatePointsToCluster "<< elapsed_secs << endl;
}


int reaccociatePointsToCluster(vector<Point> cluster_center, vector<Point>* points){
	clock_t begin = clock();
	int pointMoved = 0;
	double minDistance = 0;
	double distance = 0;
	int clusterIdx = -1;


	for (vector<Point>::iterator it = points->begin() ; it != points->end(); ++it){
		for (int i = 0 ; i < cluster_center.size(); i++){
			distance = calculateDistanceBetweenPoints(cluster_center[i], *it);

			if (i == 0)
				minDistance = distance;

			if (minDistance >= distance){
				minDistance = distance;
				clusterIdx = i;
			}
		}
		if ((*it).center_id != clusterIdx){
			(*it).center_id = clusterIdx;
			pointMoved += 1;
		}
	}
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	cout << "Time to execute reaccociatePointsToCluster "<< elapsed_secs << endl;
	return pointMoved;
}

//Get all points associated to specific clusted and calculate new cluster center - avg
void calculateClusterCenter(vector<Point> * cluster_centers, vector<Point> points){
	clock_t begin = clock();
	int i = 0;
	for (vector<Point>::iterator it = cluster_centers->begin() ; it != cluster_centers->end(); ++it){
		for (int j = 0 ; j < points[0].coordinates.size(); j++)
		{
			double coordinate_sum = 0;
			int num_of_points = 0;
			for (int k = 0 ; k < points.size() ; k++)
			{
				if (points[k].center_id == i)
				{
					num_of_points ++;
					coordinate_sum += points[k].coordinates[j];	
				}
				
			}
			(*it).coordinates[j] = coordinate_sum/num_of_points;
		}
		i++;
	}

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	cout << "Time to execute calculateClusterCenter "<< elapsed_secs << endl;

}


//Caclulate distance between 2 point accoring to the formula distance = sqrt((x1 - x2)2 + (y1 - y2)2......)
double calculateDistanceBetweenPoints(Point point1, Point point2){

	clock_t begin = clock();



	int tid;
	double distance = 0;
	double sum = 0;
	double distance_total = 0;

	//###################OMP

	//Size num of procs
	//double ompSharedArray[8] = {0};

	////omp_set_num_threads(8);

	//#pragma omp parallel for private(tid) 
	//for (int i = 0 ; i < point1.coordinates.size() ; i++)
	//{
	//	tid = omp_get_thread_num();
	//	//cout << endl <<  "*******OMP thread " << tid << endl;
	//	//Concurrency::wait(10);
	//	 
	//	ompSharedArray[tid] += pow(point1.coordinates[i] - point2.coordinates[i],2);

	//}

	//
	//for (int i = 0; i < 8; i++)
	//	sum += ompSharedArray[i];

	//distance_total = sqrt(sum);

	//###########################

	/*for (int i = 0 ; i < point1.coordinates.size() ; i++)
	{
		distance += pow(point1.coordinates[i] - point2.coordinates[i],2);
	}

	distance_total = sqrt(distance);*/
	



	//########################9
	//double* arr1 = new double[52];
	double arr1[52] = { 0 };
	makeArray(arr1, point1.coordinates);
	//arr1 = &point1.coordinates[0];
	//double* arr2 = new double[52];
	double arr2[52] = { 0 };
	makeArray(arr2, point2.coordinates);
	//arr2 = &point2.coordinates[0];
	double result = 0;
	int numOfThreads = point1.coordinates.size();
	double arr3[52] = { 0 };
	
	//CudaCalcDistance(arr1, arr2, point1.coordinates.size(), &result, numOfThreads);
	calcDistanceCoordiantesWithCuda(arr1, arr2, arr3, 52);


	double sum1 = 0;
	for (int i = 0; i < 52; i++){
		sum1 +=arr3[i];
	}


	//#########################
	

	/*cout << "distance between " <<endl;
	point1.toString() ;
	cout << endl << " and " << endl;
	point2.toString() ;
	cout << distance_total << endl;*/

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	cout << "Time to execute calculateDistanceBetweenPoints "<< elapsed_secs << endl;

	//return distance_total;

	//return result;



	return sqrt(sum1);

	
	

}


double* makeArray(double * arr, vector<double> coordinates){
	double array1[52] = {0};
	for (int i = 0; i < coordinates.size(); i++)
		arr[i] = (float)coordinates[i];
	//arr = array1;
	return arr;
}

//The diameter of a cluster is the maximum distance between any two points of the cluster
double calculateClusterDiameterOLD(vector<Point> points, vector<Point> cluster_centers, int clusterIdx){
	clock_t begin = clock();
	double max_distance_between_points_in_cluster = 0;
	for (int i = 0; i < points.size() ; i++)
	{
		if (points[i].center_id == clusterIdx)
		{
			double current_point_distance = calculateDistanceBetweenPoints(points[i],cluster_centers[clusterIdx]);
			if (current_point_distance > max_distance_between_points_in_cluster)
			{
				max_distance_between_points_in_cluster = current_point_distance;
			}	
		}
	}
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	cout << "Time to execute calculateClusterDiameterOLD "<< elapsed_secs << endl;

	return max_distance_between_points_in_cluster;
}


//The diameter of a cluster is the maximum distance between any two points of the cluster
double calculateClusterDiameter(vector<Point> points, vector<Point> cluster_centers, int clusterIdx){
	clock_t begin = clock();

	double max_distance_between_points_in_cluster = 0;
	double distance = 0;

	vector<Point> pointsInCluster;
	findAllPointsInCluster(points, &pointsInCluster, clusterIdx);

	int tid, i , j;
	double maxDistance[8] = {0};

	omp_set_num_threads(8);
	#pragma omp parallel for private(tid, distance)
	for (int i = 0; i < pointsInCluster.size(); i++){
		tid = omp_get_thread_num();

		for (int j = 0; j < pointsInCluster.size(); j++){
			if (i != j){
				distance = calculateDistanceBetweenPoints(pointsInCluster[i], pointsInCluster[j]);
				if (distance > maxDistance[tid])
			{
				maxDistance[tid] = distance;
			}	
			}
		}

	}

	//Find max
	for (int i = 0 ; i < 8; i++){
		if (max_distance_between_points_in_cluster < maxDistance[i])
			max_distance_between_points_in_cluster = maxDistance[i];
	}




	/*for (int i = 0; i < pointsInCluster.size(); i++){
		for (int j = 0; j < pointsInCluster.size(); j++){
			if (i != j){
				distance = calculateDistanceBetweenPoints(pointsInCluster[i], pointsInCluster[j]);
				if (distance > max_distance_between_points_in_cluster)
			{
				max_distance_between_points_in_cluster = distance;
			}	
			}
		}*/

	//}
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	cout << "Time to execute calculateClusterDiameter "<< elapsed_secs << endl;

	return max_distance_between_points_in_cluster;
}


void findAllPointsInCluster(vector<Point> points, vector<Point> * pointsInCluster, int clusterIdx){
	clock_t begin = clock();

	//vector<Point> local;
	//#pragma omp parallel for
	for (int i = 0; i < points.size(); i++){
		if (points[i].center_id == clusterIdx)
			//local.push_back(points[i]);
			pointsInCluster->push_back(points[i]);

	}

	/*for (int i = 0 ; i < local.size(); i++){
		pointsInCluster->push_back(local[i]);
	}
*/

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	cout << "Time to execute findAllPointsInCluster "<< elapsed_secs << endl;
}



//Caclulate QM = (d1/D12 + d1/D13 + d2/D21 + d2/D23 + d3/D31 + d3/D32) / 6 where di is a diameter of cluster i and Dij is a distance between centers of cluster i and cluster j.
double calculateQualityMeature(vector<Point> points, vector<Point> clusterCenters){
	clock_t begin = clock();
	int numOfClusters = clusterCenters.size();
	double diameter;
	double distance;
	int numOfPermutations = 0;
	double QM = 0;

	for (int i = 0; i < numOfClusters; i++){
		diameter = calculateClusterDiameter(points,clusterCenters, i);

		for (int j = 0; j < numOfClusters; j++){
			//Calculate distance between 2 cluster centers
			distance = calculateDistanceBetweenPoints(clusterCenters[i], clusterCenters[j]);
			//When distance is 0 - it means that we calculated distance to same point
			if (distance != 0){
				QM +=diameter/distance;
				numOfPermutations++;
			}
		}
	}
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	cout << "Time to execute calculateQualityMeature "<< elapsed_secs << endl;
	return QM/numOfPermutations;
}