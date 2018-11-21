/**
 * @file	helper.cpp
 *
 * @brief	Collection of helper functions
 *
 *			This namespace contains functions for matrix manipulation
 *
 * @author	Markus Wende
 * @version 1.0
 * @date	2017-2018
 * @bug		No known bugs.
 */

#include "helper.h"

using namespace std;
//using namespace essentia;

namespace helper {
void matrix_to_normalized_matrix(string path, vector<vector<double>> mSpectrum, vector<vector<double>>& m)
{
	/// get filename by removing the path and file extension
	size_t found = path.find_last_of("/\\");
	string filename = path.substr(found+1);
	found = filename.find_last_of(".");
	filename = filename.substr(0,found);

	/// get input matrix dimensions
	unsigned int timeLength = mSpectrum.size();
	unsigned int freqLength = mSpectrum[0].size()-1;

	//printf("Filename: %s\n", filename.c_str());
	//printf("Spectrum time length in #samples: %d\n", timeLength);
	//printf("Spectrum frequence length in #bands: %d\n", freqLength);

	/// maximum value initilization
	double maxValue = 0;

	/// search for the maximal value in input matrix
	for (int i = 1; i < freqLength; i++) {
		for (int j = 0; j < timeLength; j++) {
			if (mSpectrum[j][i] > maxValue)
				maxValue = mSpectrum[j][i];
		}
	}

	/// normalize every matrix value by dividing the value by the maximal value
	for (int i = 1; i < freqLength; i++) {
		for (int j = 0; j < timeLength; j++) {
			mSpectrum[j][i] = mSpectrum[j][i] / maxValue;
			m[freqLength-i][j] = mSpectrum[j][i];
		}
	}
}

void matrix_to_normalized_matrix(vector<vector<double>> &mIn, vector<vector<double>> &mOut)
{
	/// get spectrogram dimensions
	unsigned int rowSize = mIn.size();
	unsigned int columnLength = mIn[0].size();

	/// maximum value initilization
	double maxVal = 0.0;
	double minVal = 0.0;
	
	/// search for the maximal value in input matrix
	for (int i = 0; i < rowSize; i++)
	{
		for (int j = 0; j < columnLength; j++)
		{
			if (mIn.at(i).at(j) > maxVal)
				maxVal = mIn.at(i).at(j);
			if (mIn.at(i).at(j) < minVal)
				minVal = mIn.at(i).at(j);
		}
	}

	double absMin = abs(minVal);
	double absMax = absMin + maxVal;

	/// normalize every matrix value by dividing the value by the maximal value
	for (int i = 0; i < rowSize; i++)
	{
		for (int j = 0; j < columnLength; j++)
		{
			mOut.at(i).at(j) = mIn.at(i).at(j) + absMin;
			mOut.at(i).at(j) = mOut.at(i).at(j) / absMax;
		}
	}
}

void zero_mean(vector<vector<double>> &mIn, vector<vector<double>> &mOut)
{
	/// get spectrogram dimensions
	unsigned int rowSize = mIn.size();
	unsigned int columnLength = mIn[0].size();

	/// normalize every matrix value by dividing the value by the maximal value
	for (int i = 0; i < rowSize; i++)
	{
		for (int j = 0; j < columnLength; j++)
		{
			mOut.at(i).at(j) = 2 * (mIn.at(i).at(j) - 0.5);
		}
	}
}

void matrix_to_normalized_matrix(vector<vector<float>> &mIn, vector<vector<float>> &mOut)
{
	/// get spectrogram dimensions
	unsigned int rowSize = mIn.size();
	unsigned int columnLength = mIn[0].size();

	/// maximum value initilization
	float maxVal = 0.0;
	float minVal = 0.0;
	
	/// search for the maximal value in input matrix
	for (int i = 0; i < rowSize; i++)
	{
		for (int j = 0; j < columnLength; j++)
		{
			if (mIn.at(i).at(j) > maxVal)
				maxVal = mIn.at(i).at(j);
			if (mIn.at(i).at(j) < minVal)
				minVal = mIn.at(i).at(j);
		}
	}

	double absMin = abs(minVal);
	double absMax = absMin + maxVal;

	/// normalize every matrix value by dividing the value by the maximal value
	for (int i = 0; i < rowSize; i++)
	{
		for (int j = 0; j < columnLength; j++)
		{
			mOut.at(i).at(j) = mIn.at(i).at(j) + absMin;
			mOut.at(i).at(j) = mOut.at(i).at(j) / absMax;
		}
	}
}

void matrix_to_normalized_vector(vector<vector<double>> mSpectrum, unsigned int& height, unsigned int& width, vector<double>& v)
{
	/// get spectrogram dimensions
	unsigned int timeLength = mSpectrum.size();
	unsigned int freqLength = mSpectrum[0].size();

	/// save number of frequency bins in height and number of time samples in width
	height = freqLength-1;
	width = timeLength;

	/// initliaze maximal input matrix value
	double maxValue = 0;

	///	search for maximal value in input matrix
	for (int i = 1; i < freqLength; i++) {
		for (int j = 0; j < timeLength; j++) {
			if (mSpectrum[j][i] > maxValue)
				maxValue = mSpectrum[j][i];
		}
	}

	/// normalize each value of the input matrix and push him to the end of the output vector
	for (int i = 1; i < freqLength; i++) {
		for (int j = 0; j < timeLength; j++) {
			v.push_back(mSpectrum[j][i] / maxValue);
		}
	}
}

void matrix_to_vector(vector<vector<double>> mIn, unsigned int& height, unsigned int& width, vector<double>& vOut)
{
	/// get spectrogram dimensions
	unsigned int timeLength = mIn.size();
	unsigned int freqLength = mIn[0].size();

	/// save number of frequency bins in height and number of time samples in width
	height = freqLength-1;
	width = timeLength;
	
	/// normalize each value of the input matrix and push him to the end of the output vector
	for (int i = 1; i < freqLength; i++) {
		for (int j = 0; j < timeLength; j++) {
			vOut.push_back(mIn[j][i]);
		}
	}
}

void matrix_enlarge(std::vector<std::vector<double>> mInput, std::vector<std::vector<double>>& mOutput)
{
	/// initialize an extern counter to adress the different rows of the input matrix
	/// initialize maximal value inside the input matrix
	/// initialize blocksize, which represents the row multiplicator
	int counter = 0;
	double maxVal = 0.0;
	int blockSize = (int) floor(mOutput[0].size()/mInput[0].size());

	/// copy blocksize times the rows of the input matrix to the output matrix
	for (int i = 1; i < mOutput[0].size(); i++) {
		for (int j = 0; j < mOutput.size(); j++) {
			mOutput[j][i] = mInput[j][counter];
			if (mOutput[j][i] > maxVal) {maxVal = mOutput[j][i];}
		}
		/// switch to the next row of the input matrix if the blocksize is reached
		if (i > 1 && counter < (mInput[0].size()-1) && i % blockSize == 0) {
			counter++;
		}
	}
	//printf("\n=============================================================================\n");
}

vector<vector<double>> matrix_add(vector<vector<double>> A, vector<vector<double>> B)
{
	/// initliaze output matrix with zeros, dimensions of the matrix are the same as
	/// the dimensions of the input matrices
	vector<vector<double>> out(A.size(), vector<double> (A.at(0).size(), 0));

	/// loop every matrix element and add A_i_j and B_i_j up
	for (int i = 0; i < A.size(); i++)
	{
		for (int j = 0; j < A.at(0).size(); j++)
		{
			out.at(i).at(j) = A.at(i).at(j) + B.at(i).at(j);
		}
	}

	return out;
}

vector<vector<double>> matrix_add_with_const(vector<vector<double>> A, vector<vector<double>> B, double x)
{
	/// initliaze output matrix with zeros, dimensions of the matrix are the same as
	/// the dimensions of the input matrices
	vector<vector<double>> out(A.size(), vector<double> (A.at(0).size(), 0));

	/// loop every matrix element and add A_i_j and x multiplied with B_i_j up
	for (int i = 0; i < A.size(); i++)
	{
		for (int j = 0; j < A.at(0).size(); j++)
		{
			out.at(i).at(j) = A.at(i).at(j) + x * B.at(i).at(j);
		}
	}

	return out;
}

vector<vector<double>> matrix_mult(vector<vector<double>> A, vector<vector<double>> B)
{
	/// check if matrices dimensions correspond to the matrix multiplication rule
	/// A with m x n and B with n x p to get C with m x p
	if (A.at(0).size() != B.size())
	{
		cout << "Matrices cant be multiplied." << endl;
	}

	/// initliaze output matrix with zeros
	vector<vector<double>> out(A.size(), vector<double> (B.at(0).size(), 0));

	/// loop every element of the output matrix
	/// every row
	for (int i = 0; i < A.size(); i++)
	{
		/// every column
		for (int j = 0; j < B.at(0).size(); j++)
		{
			/// sum up the row of A element wise multiplied by the column of B
			for (int k = 0; k < B.size(); k++)
			{
				out.at(i).at(j) = A.at(i).at(k) * B.at(k).at(j);
			}
		}
	}

	return out;
}

vector<vector<double>> matrix_T(vector<vector<double>> A)
{
	/// initliaze output matrix with zeros
	vector<vector<double>> out(A.at(0).size(), vector<double> (A.size(), 0));

	/// loop every element of the output matrix
	for (int i = 0; i < out.size(); i++)
	{
		for (int j = 0; j < out.at(0).size(); j++)
		{
			out.at(i).at(j) = A.at(j).at(i);
		}
	}

	return out;
}

double matrix_sum(vector<vector<double>> A)
{
	double sum = 0;
	for (int i = 0; i < A.size(); i++)
	{
		for (int j = 0; j < A.at(0).size(); j++)
		{
			sum += abs(A.at(i).at(j));
		}
	}

	return sum;
}

vector<double> vec_matrix_mult(vector<double> a, vector<vector<double>> B)
{
	/// check if matrices dimensions correspond to the matrix multiplication rule
	/// A with m x n and B with n x p to get C with m x p
	if (a.size() != B.size())
	{
		cout << "Vector and matrix cant be multiplied." << endl;
	}

	/// initliaze output vector with zeros
	vector<double> out(B.at(0).size(), 0);

	/// loop every element of the output vector
	for (int i = 0; i < out.size(); i++)
	{
		/// sum up the elements in a multiplied by the elements in the column of B
		for (int j = 0; j < a.size(); j++)
		{
			out.at(i) += a.at(j) * B.at(j).at(i);
		}
	}

	return out;
}

vector<double> vec_ele_add(vector<double> a, vector<double> b)
{
	/// check if vector dimensions are the same
	if (a.size() != b.size())
	{
		cout << "Vectors cant be added element wise." << endl;
	}

	/// initliaze output vector with zeros
	vector<double> out(a.size(), 0);

	/// loop every element of the output vector
	for (int i = 0; i < out.size(); i++)
	{
		out.at(i) = a.at(i) + b.at(i);
	}

	return out;
}

vector<double> vec_ele_add_with_const(vector<double> a, vector<double> b, double C)
{
	/// check if vector dimensions are the same
	if (a.size() != b.size())
	{
		cout << "Vectors cant be added element wise." << endl;
	}

	/// initliaze output vector with zeros
	vector<double> out(a.size(), 0);

	/// loop every element of the output vector
	for (int i = 0; i < out.size(); i++)
	{
		out.at(i) = a.at(i) + C * b.at(i);
	}

	return out;
}

vector<double> vec_ele_sub(vector<double> a, vector<double> b)
{
	/// check if vector dimensions are the same
	if (a.size() != b.size())
	{
		cout << "Vectors cant be added element wise." << endl;
	}

	/// initliaze output vector with zeros
	vector<double> out(a.size(), 0);

	/// loop every element of the output vector
	for (int i = 0; i < out.size(); i++)
	{
		out.at(i) = a.at(i) - b.at(i);
	}

	return out;
}

vector<double> vec_ele_mult(vector<double> a, vector<double> b)
{
	/// check if vector dimensions are the same
	if (a.size() != b.size())
	{
		cout << "Vectors cant be multiplied element wise." << endl;
	}

	/// initliaze output vector with zeros
	vector<double> out(a.size(), 0);

	/// loop every element of the output vector
	for (int i = 0; i < out.size(); i++)
	{
		out.at(i) = a.at(i) * b.at(i);
	}

	return out;
}

vector<vector<double>> outer(vector<double> a, vector<double> b)
{
	/// initialize the output matrix for the outer product of vector a and b
	/// dimension of the output matrix is m x n, with length m of vector a and length n of
	/// vector b
	vector<vector<double>> out(a.size(), vector<double> (b.size(), 0));

	/// loop every element of the output matrix out_i_j and assign the product a_i * b_j
	for (int i = 0; i < a.size(); i++)
	{
		for (int j = 0; j < b.size(); j++) {
			out.at(i).at(j) = a.at(i) * b.at(j);
		}
	}

	return out;
}

vector<double> get_oneHot(vector<double> x)
{
	vector<double> out(x.size(), 0.0);
	int index = 0;
	double maxVal = 0.0;

	for (int i = 0; i < x.size(); i++)
	{
		if (x.at(i) >= maxVal)
		{
			maxVal = x.at(i);
			index = i;
		}
	}

	out.at(index) = 1.0;

	return out;
}

vector<double> vec_concat(vector<double> a, vector<double> b)
{
	vector<double> out(a.size() + b.size(), 0);
	int index = 0;
	
	for (int i = 0; i < a.size(); i++)
	{
		out.at(i) = a.at(i);
		index = i;
	}

	for (int j = 0; j < b.size(); j++)
	{
		out.at(index + j) = b.at(j);
	}

	return out;
}

void print_matrix(vector<vector<double>> &mIn)
{
	/// get spectrogram dimensions
	unsigned int rowSize = mIn.size();
	unsigned int columnLength = mIn[0].size();


	/// maximum value initilization
	double maxVal = 0.0;
	double minVal = 0.0;
	
	/// search for the maximal value in input matrix
	for (int i = 0; i < rowSize; i++)
	{
		for (int j = 0; j < columnLength; j++)
		{
			cout << mIn.at(i).at(j) << ",";
			if (mIn.at(i).at(j) > maxVal)
				maxVal = mIn.at(i).at(j);
			if (mIn.at(i).at(j) < minVal)
				minVal = mIn.at(i).at(j);
		}
		cout << endl;
	}
	cout << "MaxVal: " << maxVal << "\tMinVal: " << minVal << endl;
	cout << rowSize << endl;
	cout << columnLength << endl;
}

void print_2matrices_column(string label, vector<vector<double>> &mIn, vector<vector<double>> &mIn2)
{
	/// get spectrogram dimensions
	unsigned int rowSize = mIn.size();
	unsigned int columnLength = mIn.at(0).size();
	unsigned int columnLength2 = mIn2.at(0).size();


	/// maximum value initilization
	double maxVal = 0.0;
	double minVal = 0.0;
	double maxVal2 = 0.0;
	double minVal2 = 0.0;

	///
	cout << label << endl;
	cout << "====================================================================" << endl;
	/// search for the maximal value in input matrix
	for (int i = 0; i < rowSize; i++)
	{
		for (int j = 0; j < columnLength; j++)
		{
			std::cout << std::fixed;
			cout << std::setprecision(3) << mIn.at(i).at(j) << "\t";
			if (mIn.at(i).at(j) > maxVal)
				maxVal = mIn.at(i).at(j);
			if (mIn.at(i).at(j) < minVal)
				minVal = mIn.at(i).at(j);
		}
		
		cout << "\t||\t";
		for (int j = 0; j < columnLength2; j++)
		{
			std::cout << std::fixed;
			cout << std::setprecision(3) << mIn2.at(i).at(j) << "\t";
			if (mIn2.at(i).at(j) > maxVal2)
				maxVal2 = mIn2.at(i).at(j);
			if (mIn2.at(i).at(j) < minVal2)
				minVal2 = mIn2.at(i).at(j);
		}
		cout << endl;
	}
	cout << "--------------------------------------------------------------------" << endl;
	cout << "MaxVal: " << maxVal << "\tMinVal: " << minVal << "\t||\t";
	cout << "MaxVal: " << maxVal2 << "\tMinVal: " << minVal2 << endl;
	cout << "Num Rows: " << rowSize << endl;
	cout << "Num Columns mIn1: " << columnLength << "\tNum Columns min2: " << columnLength2 << endl;
	cout << "====================================================================" << endl << endl;
}

void print_matrix(string label, vector<vector<double>> &mIn)
{
	/// get spectrogram dimensions
	unsigned int rowSize = mIn.size();
	unsigned int columnLength = mIn[0].size();


	/// maximum value initilization
	double maxVal = 0.0;
	double minVal = 0.0;

	///
	cout << label << endl;
	cout << "====================================================================" << endl;
	/// search for the maximal value in input matrix
	for (int i = 0; i < rowSize; i++)
	{
		for (int j = 0; j < columnLength; j++)
		{
			std::cout << std::fixed;
			cout << std::setprecision(3) << mIn.at(i).at(j) << "\t";
			if (mIn.at(i).at(j) > maxVal)
				maxVal = mIn.at(i).at(j);
			if (mIn.at(i).at(j) < minVal)
				minVal = mIn.at(i).at(j);
		}
		cout << endl;
	}
	cout << "--------------------------------------------------------------------" << endl;
	cout << "MaxVal: " << maxVal << "\tMinVal: " << minVal << endl;
	cout << "Num Rows: " << rowSize << endl;
	cout << "Num Columns: " << columnLength << endl;
	cout << "====================================================================" << endl << endl;
}

void print_vector(string label, vector<double> &vIn)
{
	cout << label << " ";
	for (unsigned i = 0; i < vIn.size(); ++i) {
		cout << fixed << vIn[i] << " ";
	}

	cout << endl;
}

void get_textGrid_targetVals_vc(item_c& tgItem, int frame, vector<double>& targetVals)
{
	/// Train the net what the outputs should have been
	targetVals.clear();
	if (tgItem.interval[frame].text.compare("sil") == 0)
	{
		targetVals.push_back(1.0);
		targetVals.push_back(0.0);
		targetVals.push_back(0.0);
		//cout << "out: 0.0" << endl;
	} else if (tgItem.interval[frame].text.compare("c") == 0)
	{	
		targetVals.push_back(0.0);
		targetVals.push_back(1.0);
		targetVals.push_back(0.0);
		//cout << "out: 0.0" << endl;
	} else if (tgItem.interval[frame].text.compare("v") == 0)
	{	
		targetVals.push_back(0.0);
		targetVals.push_back(0.0);
		targetVals.push_back(1.0);
		//cout << "out: 1.0" << endl;
	}
}

void get_textGrid_frame(item_c& tgItem, int mIndex, int& frame, double& frameEnd, int nSamples)
{
	double time = 0;
	double multiplicator = 0;
	
	multiplicator = ((double) (mIndex+1) / (double) nSamples);
	time = tgItem.xmax * multiplicator;
	
	if (time >= frameEnd)
	{
		frameEnd = tgItem.interval[frame+1].xmax;
		frame++;
	}
		
}

void convert_float_to_double(vector<vector<float>> &in, vector<vector<double>> &out)
{
	for (int i = 0; i < in.size(); i++)
	{
		for (int j = 0; j < in.at(0).size(); j++)
		{
			out.at(i).at(j) = (double)in.at(i).at(j);
		}
	}
}

}
