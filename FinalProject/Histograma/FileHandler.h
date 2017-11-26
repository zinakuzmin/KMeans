#pragma once
#include <string>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#define readFile FileHandleReadFile
#define WriteToFile FileHandleWriteToFile
//#include "Kmeans.h"


using namespace std; 

struct Point {
	int center_id;
	int product_number;
	vector<double> coordinates;
	
	Point(vector<double> c, int pn) {
		coordinates.resize(c.size());
		coordinates = c;
		product_number = pn;
		center_id = 0;
	}

	Point() {
		center_id = 0;
	}

	void toString()
	{
		cout<<"Center_ID:"<<center_id<<endl;
		cout<<"Product Number:"<<product_number<<endl;

		for (int i = 0; i< coordinates.size() ;i++)
		{
			cout<<" ";
			cout<<coordinates[i];
		}
		cout<<endl;
	}
};

//extern FileHeader* readFile(char rout[]);
//extern FileHeader* readFile();

void readFromFile(char filepath[], int *N, int *dimentions, int *MAX, int *LIMIT, double *QM, vector<Point> * points);
//void readFromFile(char filepath[], int N, int dimentions, int MAX, int LIMIT, double QM, vector<Point> * points);
void writeResultsToFile(char const path[], vector<Point> cluster_center, float qm , int number_of_clusters);
//extern void WriteToFile(char rout[], KmeansAns *ans, long numClusters);
