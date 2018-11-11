/**
 * @file	dataset.cpp
 * @class	DATASET
 *
 * @brief	DataSet is a class fro managing the training and test sets for the neural network
 *
 *			-
 *
 * @note	-
 *
 * @author	Markus Wende
 * @version 1.0
 * @date	2017-2018
 * @bug		No known bugs.
 */

#include "dataset.h"

using namespace std;

DataSet::DataSet(const string filename)
{
	_file.open(filename.c_str());
}

void DataSet::return_to_begin_of_file()
{
	_file.clear();
	_file.seekg(0, std::ios::beg);
}

int DataSet::size()
{
	int size = 0;
	size = std::count(std::istreambuf_iterator<char>(_file),
			std::istreambuf_iterator<char>(), '\n');
	size = size / 2;
	return size;
}

void DataSet::init_set(int T, vector<unsigned> topo,
		vector<vector<double>> &X, vector<vector<double>> &Y,
		vector<vector<double>> &bX, vector<vector<double>> &bY)
{
	X.clear();
	X.resize(T, vector<double> (topo.at(0), 0));
	Y.clear();
	Y.resize(T, vector<double> (topo.at(2), 0));
	bX.clear();
	bX.resize(T, vector<double> (topo.at(0), 0));
	bY.clear();
	bY.resize(T, vector<double> (topo.at(2), 0));

	int t = 0;
	while (t < T)
	{
		string line;
		getline(_file, line);
		stringstream ss(line);

		string label;
		ss >> label;

		if (label.compare("in:") == 0)
		{
			double oneValue;
			int element = 0;
			while (ss >> oneValue)
			{
				X.at(t).at(element) = oneValue;
				element++;
			}
		} else if (label.compare("out:") == 0)
		{
			double oneValue;
			int element = 0;
			while (ss >> oneValue)
			{
				Y.at(t).at(element) = oneValue;
				element++;
			}
			t++;
		}	
	}

	bX.at(0) = X.at(T-1);
	bY.at(0) = Y.at(T-1);
	
	t = 1;
	while (t < T)
	{
		string line;
		getline(_file, line);
		stringstream ss(line);

		string label;
		ss >> label;

		if (label.compare("in:") == 0)
		{
			double oneValue;
			int element = 0;
			while (ss >> oneValue)
			{
				bX.at(t).at(element) = oneValue;
				element++;
			}
		} else if (label.compare("out:") == 0)
		{
			double oneValue;
			int element = 0;
			while (ss >> oneValue)
			{
				bY.at(t).at(element) = oneValue;
				element++;
			}
			t++;
		}	
	}
}

void DataSet::shift_set(int steps, vector<vector<double>> &X, vector<vector<double>> &Y,
		vector<vector<double>> &bX, vector<vector<double>> &bY)
{
	vector<vector<double>> XCopy;
	vector<vector<double>> YCopy;
	vector<vector<double>> bXCopy;
	vector<vector<double>> bYCopy;

	copy(X.begin(), X.end(), back_inserter(XCopy));
	copy(Y.begin(), Y.end(), back_inserter(YCopy));
	copy(bX.begin(), bX.end(), back_inserter(bXCopy));
	copy(bY.begin(), bY.end(), back_inserter(bYCopy));

	X.clear();
	X.resize(XCopy.size(), vector<double> (XCopy.at(0).size(), 0));
	Y.clear();
	Y.resize(YCopy.size(), vector<double> (YCopy.at(0).size(), 0));
	bX.clear();
	bX.resize(bXCopy.size(), vector<double> (bXCopy.at(0).size(), 0));
	bY.clear();
	bY.resize(bYCopy.size(), vector<double> (bYCopy.at(0).size(), 0));

	for (int row = 0; row < X.size() - steps; row++)
	{
		for (int col = 0; col < X.at(0).size(); col++)
		{
			X.at(row).at(col) = XCopy.at(row+steps).at(col);
			bX.at(row).at(col) = bXCopy.at(row+steps).at(col);
		}
		
		for (int col = 0; col < Y.at(0).size(); col++)
		{
			Y.at(row).at(col) = YCopy.at(row+steps).at(col);	
			bY.at(row).at(col) = bYCopy.at(row+steps).at(col);	
		}
	}

	int start = X.size() - steps;

	for (int i = 0; i < steps; i++)
	{
		X.at(start+i) = bXCopy.at(i+1);
		Y.at(start+i) = bYCopy.at(i+1);

		bool done = false;
		while(!done)
		{
			string line;
			getline(_file, line);
			stringstream ss(line);

			string label;
			ss >> label;

			if (label.compare("in:") == 0)
			{
				double oneValue;
				int element = 0;
				while (ss >> oneValue)
				{
					bX.at(start+i).at(element) = oneValue;
					element++;
				}
			} else if (label.compare("out:") == 0)
			{
				double oneValue;
				int element = 0;
				while (ss >> oneValue)
				{
					bY.at(start+i).at(element) = oneValue;
					element++;
				}
				done = true;;
			}
		}	
	}
}
