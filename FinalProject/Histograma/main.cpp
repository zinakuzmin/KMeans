// 327366746 Zina Kuzmin
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <ctime>
#include <concrt.h>

#include "FileHandler.h"
#include "kernel.h"
#include "kmeans.h"



void printPoints(vector<Point> points);
void printPointsArray(Point * points, int size);

void countPointsInCluster1(int numOfCenters, Point * points, int numOfPoints);
int countPointsInClusterID(int clusterNumber, Point * points, int numOfPoints);

void convertToArray(vector<Point> points, Point * arrPoints);
void convertToVector(vector<Point> * points, Point * arrPoints, int numOfPoints);

double runParallelProgram(Point * cluster_centers, Point * pointsArray, int numOfPoints, int numOfCoordinates, double QM, int LIMIX, int MAX, int number_of_clusters);
double runSequentialProgram(Point * cluster_centers, Point * pointsArray, int numOfPoints, int numOfCoordinates, double QM, int LIMIT, int MAX, int number_of_clusters);


int main(int argc,char *argv[]){	
	int N = 0;   //Number of products
	int n = 0;   //Dimention
	int MAX = 0;  //Maximum number of clusters to find
	int LIMIT = 0; //Maximum number of iterations of kmeans run
	double QM = 0;  //Quality measure
	vector<Point> points;
	vector<Point> centers;
	char filepath[] = ("dataset_normalized.txt");
	//char filepath[] = ("testDataSet.txt");
	int number_of_clusters = INITIAL_NUM_OF_CLUSTERS;
	Point * pointsArray;
	Point * cluster_centers = NULL;
	omp_set_num_threads(OMP_THREADS);


	/* Read data from input file which is located in project folder */
	readFromFile(filepath, &N, &n, &MAX, &LIMIT, &QM, &points);
	

	/* Terminate program if input file has more than 5000 products */
	if (points.size() > MAX_NUM_OF_PRODUCTS)
		return 0;

	
	/* Convert vertor to array */
	pointsArray = new Point[N];
	convertToArray(points, pointsArray);


	/* Start time measure for program execution */
	const clock_t begin_time = clock();

	double qm = 0;
	if (RUN_PARALLEL == 1)
		qm = runParallelProgram(cluster_centers, pointsArray, N, n, QM, LIMIT, MAX, number_of_clusters);
	else
		qm = runSequentialProgram(cluster_centers, pointsArray, N, n, QM, LIMIT, MAX, number_of_clusters);


	/* Print full execution time */
	cout << "execution time " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;

	/* Free dymanic allocations */
	delete  [] pointsArray;
	delete [] cluster_centers;


	/* Write cluster centers to results file in project folder */
	convertToVector(&centers, cluster_centers, number_of_clusters-1);
	writeResultsToFile("results.txt", centers, qm, number_of_clusters-1);

	cout<<"Execution completed, press return\n" << endl;



	return 0;
}



double runSequentialProgram(Point * cluster_centers, Point * pointsArray, int numOfPoints, int numOfCoordinates, double QM, int LIMIT, int MAX, int number_of_clusters){

	double qm = 0;

	do{
		/*Allocate array for cluster centers*/
		cluster_centers = new Point[number_of_clusters];

		cout << "****************************************" << endl;
		cout << "Number of clusters = " << number_of_clusters << endl;


		/*Choose first K points as clusters centers */
		for (int i = 0;i < number_of_clusters;i++){
			cluster_centers[i] = pointsArray[i];
		}

		/*cout << "*******************Print initial centers*********************" << endl;
		printPointsArray(cluster_centers, number_of_clusters);*/

		/* Associate points to initial clusters */
		associatePointsToClusterSeq(cluster_centers, pointsArray, numOfPoints, number_of_clusters, 1);

		/* Print how many points each cluster has after assiciation */
		countPointsInCluster1(number_of_clusters, pointsArray, numOfPoints);


		int pointMoved = 0;
		int iteration = 0;

		do{
			iteration++;

			/* Calculate cluster center as average of points that belong to this cluster */
			calculateClusterCenterSeq(cluster_centers, pointsArray, numOfPoints, number_of_clusters);

			/*cout << "*******************Print centers after recalc*********************" << endl;
			printPointsArray(cluster_centers, number_of_clusters);*/


			pointMoved = associatePointsToClusterSeq(cluster_centers, pointsArray, numOfPoints, number_of_clusters, 0);
			cout << "After recalc points were moved " << pointMoved << endl;
			countPointsInCluster1(number_of_clusters, pointsArray, numOfPoints);

		}
		while (pointMoved != 0 && iteration < LIMIT);

		qm = calculateQualityMeatureSeq(pointsArray, cluster_centers, numOfPoints, number_of_clusters);
		cout << "#########calculated quality is " << qm << " and expected " << QM << "#############" << endl; 
		number_of_clusters++;
	}
	while (QM < qm && number_of_clusters <= MAX);

	return qm;
}


double runParallelProgram(Point * cluster_centers, Point * pointsArray, int numOfPoints, int numOfCoordinates, double QM, int LIMIT, int MAX, int number_of_clusters){

	double qm = 0;

	do{
		/*Allocate array for cluster centers*/
		cluster_centers = new Point[number_of_clusters];

		cout << "****************************************" << endl;
		cout << "Number of clusters = " << number_of_clusters << endl;


		/*Choose first K points as clusters centers */
		for (int i = 0;i < number_of_clusters;i++){
			cluster_centers[i] = pointsArray[i];
		}

		/*cout << "*******************Print initial centers*********************" << endl;
		printPointsArray(cluster_centers, number_of_clusters);*/

		/* Associate points to initial clusters */
		associatePointsToClusterParallel(cluster_centers, pointsArray, numOfPoints, number_of_clusters, 1);

		/* Print how many points each cluster has after assiciation */
		countPointsInCluster1(number_of_clusters, pointsArray, numOfPoints);


		int pointMoved = 0;
		int iteration = 0;

		do{
			iteration++;

			/* Calculate cluster center as average of points that belong to this cluster */
			calculateClusterCenterParralel(cluster_centers, pointsArray, numOfPoints, number_of_clusters);

			/*cout << "*******************Print centers after recalc*********************" << endl;
			printPointsArray(cluster_centers, number_of_clusters);*/


			pointMoved = associatePointsToClusterParallel(cluster_centers, pointsArray, numOfPoints, number_of_clusters, 0);
			cout << "After recalc points were moved " << pointMoved << endl;
			countPointsInCluster1(number_of_clusters, pointsArray, numOfPoints);

		}
		while (pointMoved != 0 && iteration < LIMIT);

		qm = calculateQualityMeatureParallel(pointsArray, cluster_centers, numOfPoints, number_of_clusters);
		cout << "#########calculated quality is " << qm << " and expected " << QM << "#############" << endl; 
		number_of_clusters++;
	}
	while (QM < qm && number_of_clusters <= MAX);

	return qm;
}


void convertToArray(vector<Point> points, Point * arrPoints){

	for (int i = 0; i < points.size(); i++){
		arrPoints[i] = points[i];
	}
}



void convertToVector(vector<Point> * points, Point * arrPoints, int numOfPoints){
	for (int i = 0; i < numOfPoints; i++){
		points->push_back(arrPoints[i]);

	}

}

void printPointsArray(Point * points, int size){
	clock_t begin = clock();
	cout << "##############################" << endl;
	for (int i = 0 ; i < size; i++){
		points[i].toString();
	}

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Time to execute printPoints "<< elapsed_secs << endl;

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



void countPointsInCluster1(int numOfCenters, Point * points, int numOfPoints){
	clock_t begin = clock();
	int count = 0;

	for (int j = 0; j < numOfCenters; j++){
		for (int i = 0; i < numOfPoints; i++){
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

