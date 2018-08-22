/**
 * @file		helper.h
 *
 * @brief		Collection of helper functions
 *
 *					This namespace contains functions for matrix manipulation
 *
 * @author	Markus Wende
 * @version 1.0
 * @date		2017-2018
 * @bug			No known bugs.
 */

#ifndef HELPER_H
#define HELPER_H

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

/// group functions in namespace helper
namespace helper {
	/**
	 * normalize the matrix by divide every matrix entry by the maximum value
	 * in the matrix
	 * @param	path contains the path and name of the file that is executed
	 * @param mSpectrum input matrix
	 * @param m is a reference to the normalized matrix
	 * @return void
	 */
	void matrix_to_normalized_matrix(
			std::string path,
			std::vector<std::vector<float>> mSpectrum,
			std::vector<std::vector<float>>& m
			);
	
	/**
	 * normalize the matrix by divide every matrix entry by the maximum value
	 * in the matrix
	 * @param mSpectrum input matrix
	 * @param m is a reference to the normalized matrix
	 * @return void
	 */
	void matrix_to_normalized_matrix(
			std::vector<std::vector<float>> mSpectrum,
			std::vector<std::vector<float>>& m
			);
	
	/**
	 * convert the input matrix to a normalized vector by attaching every row
	 * of the input matrix
	 * @param	mSpectrum input matrix
	 * @param height return the input matrix height = number of frequency bins
	 * @param width return the input matrix width = number of time frames
	 * @param v reference to the normalized output vector
	 * @return void
	 */
	void matrix_to_normalized_vector(
			std::vector<std::vector<float>> mSpectrum,
			unsigned int& height,
			unsigned int& width,
			std::vector<float>& v
			);
	
	/**
	 * enlarge one matrix height to the other matrix hight
	 * by duplicating the rows x times, in which x = floor( inputMatrixHeight / outputMatrixHeight)
	 * @param	mInput input matrix
	 * @param mOutput output matrix
	 * @return void
	 */
	void matrix_enlarge(
			std::vector<std::vector<float>> mInput,
			std::vector<std::vector<float>>& mOutput
			);
}
#endif		// HELPER_H
