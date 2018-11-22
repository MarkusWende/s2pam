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
#include <cmath>
#include <malloc.h>
#include <png.h>
#include <iostream>							// std::cout, std::fixed
#include <vector>
#include <fstream>
//#include <essentia/algorithmfactory.h>
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
			std::vector<std::vector<double>> mSpectrum,
			std::vector<std::vector<double>>& m
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
			std::vector<std::vector<double>> &mIn,
			std::vector<std::vector<double>> &mOut
			);
	
	void zero_mean(
			std::vector<std::vector<double>> &mIn,
			std::vector<std::vector<double>> &mOut
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
			std::vector<std::vector<double>> mSpectrum,
			unsigned int& height,
			unsigned int& width,
			std::vector<double>& v
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
			std::vector<std::vector<double>> mIn,
			unsigned int& height,
			unsigned int& width,
			std::vector<double>& vOut
			);
	
	/**
	 * enlarge one matrix height to the other matrix hight
	 * by duplicating the rows x times, in which x = floor( inputMatrixHeight / outputMatrixHeight)
	 * @param mInput input matrix
	 * @param mOutput output matrix
	 * @return void
	 */
	void matrix_enlarge(
			std::vector<std::vector<double>> mInput,
			std::vector<std::vector<double>>& mOutput
			);

	/**
	 * A + B
	 * add two matrices with each other, Dimension of A an B has to be the same
	 * @param A input matrix A, |R^(m x n)
	 * @param B input matrix B, |R^(m x n)
	 * @return vector<vector<double>> the summarized matrix, |R^(m x n)
	 */
	std::vector<std::vector<double>> matrix_add(
			std::vector<std::vector<double>> A,
			std::vector<std::vector<double>> B
			);
	
	/**
	 * A + x * B
	 * add two matrices with each other by multiplying a constant to every
	 * element of the secound matrix,
	 * Dimension of A an B has to be the same
	 * @param A input matrix A, m x n
	 * @param B input matrix B, m x n
	 * @param x constant that B is multiplied with
	 * @return vector<vector<double>> the summarized matrix, |R^(m x n)
	 */
	std::vector<std::vector<double>> matrix_add_with_const(
			std::vector<std::vector<double>> A,
			std::vector<std::vector<double>> B,
			double x
			);
	
	/**
	 * A * B
	 * multiply two matrices with each other, column size m of A has to be
	 * the same size as row size m of B
	 * @param A input matrix A, |R^(n x m)
	 * @param B input matrix B, |R^(m x p)
	 * @return vector<vector<double>> the multiplied matrix, |R^(n x p)
	 */
	std::vector<std::vector<double>> matrix_mult(
			std::vector<std::vector<double>> A,
			std::vector<std::vector<double>> B
			);

	/**
	 * A.T
	 * transpose given matrix
	 * @param A input matrix A, |R^(n x m)
	 * @return vector<vector<double>> the tranposed matrix, |R^(m x n)
	 */
	std::vector<std::vector<double>> matrix_T(
			std::vector<std::vector<double>> A
			);

	double matrix_sum(
			std::vector<std::vector<double>> A
			);
	
	/**
	 * a * B
	 * multiply a vector with a matrix, length of a has to be
	 * the same size as column size m of B
	 * @param a input vector a, |R^(1 x m)
	 * @param B input matrix B, |R^(m x p)
	 * @return vector<double> the product, |R^p
	 */
	std::vector<double> vec_matrix_mult(
			std::vector<double> a,
			std::vector<std::vector<double>> B
			);
	
	/**
	 * a + b
	 * add to vectors element wise together
	 * @param a input vector a, |R^m
	 * @param b input vector b, |R^m
	 * @return vector<double> output vector, |R^m
	 */
	std::vector<double> vec_ele_add(
			std::vector<double> a,
			std::vector<double> b
			);

	/**
	 * a + C * b
	 * add to vectors element wise together with constant multiplied with each element
	 * of the 2nd vector
	 * @param a input vector a, |R^m
	 * @param b input vector b, |R^m
	 * @param C constant that is multiplied with vector b
	 * @return vector<double> output vector, |R^m
	 */
	std::vector<double> vec_ele_add_with_const(
			std::vector<double> a,
			std::vector<double> b,
			double C
			);

	/**
	 * subtract to vectors element wise
	 * @param a input vector a, |R^m
	 * @param b input vector b, |R^m
	 * @return vector<double> output vector, |R^m
	 */
	std::vector<double> vec_ele_sub(
			std::vector<double> a,
			std::vector<double> b
			);

	/**
	 * multiply to vectors element wise together
	 * @param a input vector a, |R^m
	 * @param b input vector b, |R^m
	 * @return vector<double> output vector, |R^m
	 */
	std::vector<double> vec_ele_mult(
			std::vector<double> a,
			std::vector<double> b
			);

	/**
	 * a \otimes b
	 * outer product function of two vectors
	 * @param a input vector a, |R^m
	 * @param b input vector b, |R^n
	 * @return vector<vector<double>> the outer product matrix, |R^(m x n)
	 */
	std::vector<std::vector<double>> outer(
			std::vector<double> a,
			std::vector<double> b
			);

	std::vector<double> get_oneHot(
			std::vector<double> x
			);

	/**
	 * concatenate two vectors a and b with each other
	 * @param a input vector a, |R^m
	 * @param b input vector b, |R^n
	 * @return vector<double> concatenated vector, |R^(1 x (m+n))
	 */
	std::vector<double> vec_concat(
			std::vector<double> a,
			std::vector<double> b
			);
	
	/**
	 * print the content of the matrix
	 * @param mIn input matrix
	 * @return void
	 */
	void print_matrix(
			std::vector<std::vector<double>> &mIn
			);

	/**
	 * print the content of the matrix
	 * @param label a string cintaining something descriptive for the matrix
	 * @param mIn input matrix
	 * @return void
	 */
	void print_matrix(
			std::string label,
			std::vector<std::vector<double>> &mIn
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
			std::vector<std::vector<double>> &mIn,
			std::vector<std::vector<double>> &mIn2
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
	
	void get_textGrid_targetVals_vc(
			item_c& tgItem,
			int frame,
			std::vector<double>& targetVals);
			
	void get_textGrid_targetVals_phn(
			item_c& tgItem,
			int frame,
			std::vector<double>& targetVals);
			
	void get_textGrid_frame(
			item_c& tgItem,
			int mIndex,
			int& frame,
			double& frameEnd,
			int nSamples);
	
	void convert_float_to_double(
			std::vector<std::vector<float>> &in,
			std::vector<std::vector<double>> &out
			);

}
#endif		// HELPER_H
