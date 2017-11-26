#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>


using namespace std;



struct Point {
	int center_id;
	int product_number;
	vector<double> coordinates;
	
	Point(vector<double> c, int pn) {
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

vector<double> getVectorFromLine(string line){
	string delimiter = "\t";

	size_t pos = 0;
	int colNum = 0;
	string token;
	
	double arr[52] = {0};

	while ((pos = line.find(delimiter)) != string::npos) {
		token = line.substr(0, pos);
		//cout << token << endl;
		arr[colNum] = strtod((token).c_str(),NULL);
		colNum++;
		line.erase(0, pos + delimiter.length());
	}

	return vector<double>(arr, arr + sizeof(arr) / sizeof(arr[0]) );
}

void writePointsToFile(char const path[], vector<Point> centers, float qm , int number_of_clusters)
{
	int tmp =0;
	FILE *file = fopen(path, "w");
	fprintf(file, "K= %d QM= %f \n",number_of_clusters,qm);
	string str;
	for (int i = 0 ; i < centers.size(); i++)
	{
		str = "C";
		fprintf(file, "C%d ", i);
		str += i;
		for (int j = 0 ; j < centers[i].coordinates.size(); j++)
		{
			str += centers[i].coordinates[j];
			str += " ";
			fprintf(file, "%f ", centers[i].coordinates[j]);
		}
		str += "\n";
		fprintf(file, "\n");
		str.clear();
	}
	fclose(file);
}


int main()
{
	int N;   //Number of products
	int n;   //Dimention
	double MAX;  //Maximum number of clusters to find
	int LIMIT; //Maximum number of iterations of kmeans run
	double QM;  //Quality measure
	vector<Point> points;


    ifstream inputFile("dataset_normalized.txt");
    string line;


	int i=0;
    while (getline(inputFile, line))
    {
        istringstream ss(line);

		//cout<<line<<endl;



		if (i == 0){		
	        ss >> N >> n >> MAX >> LIMIT >> QM;
			i++;
			continue;
		}

		Point *p = new Point(getVectorFromLine(line), i-1);
		points.push_back(*p);

		i++;
    }

	writePointsToFile("results.txt", points, QM, MAX);

	cout<<"Execution completed, press return";
	getchar();
}