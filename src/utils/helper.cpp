#include "helper.h"

using namespace std;
using namespace essentia;

namespace helper {
void matrix_to_normalized_matrix(string path, vector<vector<float>> mSpectrum, vector<vector<float>>& m) {
	// get filename without path and fileending
	size_t found = path.find_last_of("/\\");
	string filename = path.substr(found+1);
	found = filename.find_last_of(".");
	filename = filename.substr(0,found);

	// get spectrogram dimensions
	unsigned int timeLength = mSpectrum.size();
	unsigned int freqLength = mSpectrum[0].size()-1;

	printf("Filename: %s\n", filename.c_str());
	//printf("Spectrum time length in #samples: %d\n", timeLength);
	//printf("Spectrum frequence length in #bands: %d\n", freqLength);

	float maxValue = 0;

	for (int i = 1; i < freqLength; i++) {
		for (int j = 0; j < timeLength; j++) {
			if (mSpectrum[j][i] > maxValue)
				maxValue = mSpectrum[j][i];
		}
	}

	for (int i = 1; i < freqLength; i++) {
		for (int j = 0; j < timeLength; j++) {
			mSpectrum[j][i] = mSpectrum[j][i] / maxValue;
			m[freqLength-i][j] = mSpectrum[j][i];
		}
	}
}

void matrix_to_normalized_vector(vector<vector<float>> mSpectrum, unsigned int& height, unsigned int& width, vector<float>& v) {

	// get spectrogram dimensions
	unsigned int timeLength = mSpectrum.size();
	unsigned int freqLength = mSpectrum[0].size();

	height = freqLength-1;
	width = timeLength;

	float maxValue = 0;

	for (int i = 1; i < freqLength; i++) {
		for (int j = 0; j < timeLength; j++) {
			if (mSpectrum[j][i] > maxValue)
				maxValue = mSpectrum[j][i];
		}
	}

	for (int i = 1; i < freqLength; i++) {
		for (int j = 0; j < timeLength; j++) {
			v.push_back(mSpectrum[j][i] / maxValue);
		}
	}
}

void matrix_enlarge(std::vector<std::vector<float>> mInput, std::vector<std::vector<float>>& mOutput) {

	E_INFO("\tEnlarge Matrix:\n" <<
			"\t\tInput <- Hoehe: " << mInput[0].size() << ", Breite: " << mInput.size() <<
			"\n\t\tOutput: -> Hoehe: " << mOutput[0].size() << ", Breite: " << mOutput.size());

	int counter = 1;
	float maxVal = 0.0;
	int blockSize = (int) floor(mOutput[0].size()/mInput[0].size());
	for (int i = 1; i < mOutput[0].size(); i++) {
		for (int j = 0; j < mOutput.size(); j++) {
			mOutput[j][i] = mInput[j][counter];
			if (mOutput[j][i] > maxVal) {maxVal = mOutput[j][i];}
		}
		if (i > 1 && counter < (mInput[0].size()-1) && i % blockSize == 0) {
			counter++;
		}
	}
	//printf("\n=============================================================================\n");
}
}
