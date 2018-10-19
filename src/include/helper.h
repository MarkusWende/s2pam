/**
 * @file	helper.h
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

#ifndef HELPER_H
#define HELPER_H

#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <png.h>
#include <iostream>							// std::cout, std::fixed
#include <vector>
#include <fstream>
#include <malloc.h>
#include <png.h>
#include <essentia/algorithmfactory.h>
#include <iomanip>							// std::setprecision

#include "textgrid.h"

/// group functions in namespace helper
namespace helper {
	/**
	 * normalize the matrix by divide every matrix entry by the maximum value
	 * in the matrix
	 * @param path contains the path and name of the file that is executed
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
	 * normalize the matrix elements to a range betwen [0,1] by adding the absolute value of the minimum
	 * value in the matrix to every element and dividing every matrix entry by the maximum value + absolute of
	 * the minimum value
	 * in the matrix
	 * @param mIn reference to the input matrix
	 * @param mOut is a reference to the normalized output matrix
	 * @return void
	 */
	void matrix_to_normalized_matrix(
			std::vector<std::vector<float>> &mIn,
			std::vector<std::vector<float>> &mOut
			);
	
	/**
	 * convert the input matrix to a normalized vector by attaching every row
	 * of the input matrix
	 * @param mSpectrum input matrix
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
	 * convert the input matrix to a normalized vector by attaching every row
	 * of the input matrix
	 * @param mIn input matrix
	 * @param height return the input matrix height = number of frequency bins
	 * @param width return the input matrix width = number of time frames
	 * @param vOut reference to the normalized output vector
	 * @return void
	 */
	void matrix_to_vector(
			std::vector<std::vector<float>> mIn,
			unsigned int& height,
			unsigned int& width,
			std::vector<float>& vOut
			);
	
	/**
	 * enlarge one matrix height to the other matrix hight
	 * by duplicating the rows x times, in which x = floor( inputMatrixHeight / outputMatrixHeight)
	 * @param mInput input matrix
	 * @param mOutput output matrix
	 * @return void
	 */
	void matrix_enlarge(
			std::vector<std::vector<float>> mInput,
			std::vector<std::vector<float>>& mOutput
			);
	
	/**
	 * print the content of the matrix
	 * @param mIn input matrix
	 * @return void
	 */
	void print_matrix(
			std::vector<std::vector<float>> &mIn
			);

	/**
	 * print the content of the matrix
	 * @param label a string cintaining something descriptive for the matrix
	 * @param mIn input matrix
	 * @return void
	 */
	void print_matrix(
			std::string label,
			std::vector<std::vector<float>> &mIn
			);

	/**
	 * print the content of the matrix
	 * @param label a string cintaining something descriptive for the matrix
	 * @param mIn input matrix 1
	 * @param mIn2 input matrix 2
	 * @return void
	 */
	void print_2matrices_column(
			std::string label,
			std::vector<std::vector<float>> &mIn,
			std::vector<std::vector<float>> &mIn2
			);

	/**
	 * print the content of the vector
	 * @param label a string containing something descriptive for the vector
	 * @param vIn input vector
	 * @return void
	 */
	void print_vector(
			std::string label,
			std::vector<double> &vIn
			);
	
	/**
	 * print the content of the vector
	 * @param label a string containing something descriptive for the vector
	 * @param vIn input vector
	 * @return void
	 */
	void print_vector(
			std::string label,
			std::vector<float> &vIn
			);

	void get_textGrid_targetVals_vc(
			item_c& tgItem,
			int frame,
			std::vector<double>& targetVals);
			
	void get_textGrid_frame(
			item_c& tgItem,
			int mIndex,
			int& frame,
			float& frameEnd,
			int nSamples);
}
#endif		// HELPER_H
