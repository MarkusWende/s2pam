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
		vector<vector<double>> &X, vector<vector<double>> &Y)
{
	X.clear();
	X.resize(T, vector<double> (topo.at(0), 0));
	Y.clear();
	Y.resize(T, vector<double> (topo.at(2), 0));

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
}

void DataSet::shift_set(int steps, vector<vector<double>> &X, vector<vector<double>> &Y)
{
	vector<vector<double>> XCopy;
	vector<vector<double>> YCopy;

	copy(X.begin(), X.end(), back_inserter(XCopy));
	copy(Y.begin(), Y.end(), back_inserter(YCopy));

	X.clear();
	X.resize(XCopy.size(), vector<double> (XCopy.at(0).size(), 0));
	Y.clear();
	Y.resize(YCopy.size(), vector<double> (YCopy.at(0).size(), 0));

	for (int col = 0; col < X.size() - steps; col++)
	{
		for (int rowX = 0; rowX < X.at(0).size(); rowX++)
		{
			X.at(col).at(rowX) = XCopy.at(col+steps).at(rowX);
		}
		
		for (int rowY = 0; rowY < Y.at(0).size(); rowY++)
		{
			Y.at(col).at(rowY) = YCopy.at(col+steps).at(rowY);	
		}
	}

	int start = X.size() - steps;

	for (int i = 0; i < steps; i++)
	{
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
					X.at(start+i).at(element) = oneValue;
					element++;
				}
			} else if (label.compare("out:") == 0)
			{
				double oneValue;
				int element = 0;
				while (ss >> oneValue)
				{
					Y.at(start+i).at(element) = oneValue;
					element++;
				}
				done = true;;
			}
		}	
	}
}
