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
using namespace essentia;

namespace helper {
void matrix_to_normalized_matrix(string path, vector<vector<float>> mSpectrum, vector<vector<float>>& m)
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
	float maxValue = 0;

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

void matrix_to_normalized_matrix(vector<vector<float>> mSpectrum, vector<vector<float>>& m)
{
	/// get spectrogram dimensions
	unsigned int timeLength = mSpectrum.size();
	unsigned int freqLength = mSpectrum[0].size()-1;

	/// maximum value initilization
	float maxValue = 0;

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

void matrix_to_normalized_vector(vector<vector<float>> mSpectrum, unsigned int& height, unsigned int& width, vector<float>& v)
{
	/// get spectrogram dimensions
	unsigned int timeLength = mSpectrum.size();
	unsigned int freqLength = mSpectrum[0].size();

	/// save number of frequency bins in height and number of time samples in width
	height = freqLength-1;
	width = timeLength;

	/// initliaze maximal input matrix value
	float maxValue = 0;

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

void matrix_enlarge(std::vector<std::vector<float>> mInput, std::vector<std::vector<float>>& mOutput)
{
	/// print matrix information, verbose mode has to be on
	E_INFO("---Enlarge Matrix:\n" <<
			"\t\tInput <- Hoehe: " << mInput[0].size() << ", Breite: " << mInput.size() <<
			"\n\t\tOutput: -> Hoehe: " << mOutput[0].size() << ", Breite: " << mOutput.size());

	/// initialize an extern counter to adress the different rows of the input matrix
	/// initialize maximal value inside the input matrix
	/// initialize blocksize, which represents the row multiplicator
	int counter = 1;
	float maxVal = 0.0;
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
}
