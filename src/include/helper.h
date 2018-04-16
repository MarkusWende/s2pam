#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <png.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <malloc.h>
#include <png.h>
#include <essentia/algorithmfactory.h>

namespace helper {
	void matrix_to_normalized_matrix(std::string path, std::vector<std::vector<float>> mSpectrum, std::vector<std::vector<float>>& m);
	void matrix_to_normalized_vector(std::vector<std::vector<float>> mSpectrum, unsigned int& height, unsigned int& width, std::vector<float>& v);
	void matrix_enlarge(std::vector<std::vector<float>> mInput, std::vector<std::vector<float>>& mOutput);
}
