#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "FileHandler.h"

using namespace std;

void readFromFile(char filepath[], int *N, int *dimentions, int *MAX, int *LIMIT, double *QM, vector<Point> * points);
//void readFromFile(char filepath[], int N, int dimentions, int MAX, int LIMIT, double QM, vector<Point>  * points);
void writeResultsToFile(char const path[], vector<Point> cluster_center, float qm , int number_of_clusters);


vector<double> getVectorFromLine(string line, int numOfColumns){
	string delimiter = "\t";

	size_t pos = 0;
	string token;
	vector<double> pointCoordinates;
	
	//cout << "Num of colunms "<< numOfColumns << endl;

	while ((pos = line.find(delimiter)) != string::npos) {
		token = line.substr(0, pos);
		pointCoordinates.push_back(strtod((token).c_str(),NULL));
		line.erase(0, pos + delimiter.length());
	}
	
	pos = line.find(delimiter);
	token = line.substr(0, pos);
	pointCoordinates.push_back(strtod((token).c_str(),NULL));
	
	
	//cout<<"\nThisIsSize:"<<pointCoordinates.size()<<endl;
	return pointCoordinates;
}

void writeResultsToFile(char const path[], vector<Point> cluster_center, float qm , int number_of_clusters)
{
	int tmp =0;
	FILE *file = fopen(path, "w");
	fprintf(file, "Number of clusters with the best measure\n");
	fprintf(file, "K = %d QM = %f \n",number_of_clusters,qm);
	fprintf(file, "Centers of the clusters:\n");
	string str;
	for (int i = 0 ; i < cluster_center.size(); i++)
	{
		str = "C";
		fprintf(file, "C%d ", i);
		str += i;
		for (int j = 0 ; j < cluster_center[i].coordinates.size(); j++)
		{
			str += cluster_center[i].coordinates[j];
			str += " ";
			fprintf(file, "%f ", cluster_center[i].coordinates[j]);
		}
		str += "\n";
		fprintf(file, "\n");
		str.clear();
	}
	fclose(file);
}


void readFromFile(char filepath[], int *N, int *dimentions, int *MAX, int *LIMIT, double *QM, vector<Point> * points){
	

    ifstream inputFile(filepath);
    string line;


	int i=0;
    while (getline(inputFile, line))
    {
        istringstream ss(line);


		/*cout << "********Print line input " << i << endl;
		cout<<line<<endl;*/



		if (i == 0){		
	        ss >> *N >> *dimentions >> *MAX >> *LIMIT >> *QM;
			//cout <<  *N << *dimentions << *MAX << *LIMIT << *QM;
			i++;
			
			//(*points).resize(N);
			continue;
		}

		Point *p = new Point(getVectorFromLine(line, *dimentions), i-1);

		//p->toString();
		points->push_back(*p);


		i++;
    }
	//cout <<"mashu";

}

//int main()
//{
//	int N;   //Number of products
//	int n;   //Dimention
//	double MAX;  //Maximum number of clusters to find
//	int LIMIT; //Maximum number of iterations of kmeans run
//	double QM;  //Quality measure
//	vector<Point> points;
//
//	char filepath[] = ("dataset_normalized.txt");
//	readFromFile(filepath, N, n, MAX, LIMIT, QM, points);
//
//	writeResultsToFile("results.txt", points, QM, MAX);
//
//	cout<<"Execution completed, press return";
//	getchar();
//}