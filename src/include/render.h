/**
 * @file	render.h
 *
 * @brief	Collection of render functions mostly for generating files from matrices
 *
 *			This namespace contains functions mostly for file generation, e.g.
 *			png's from matrices or Mel Frequency Cepstral Coefficients storage files.
 *
 * @author	Markus Wende
 * @version 1.0
 * @date	2017-2018
 * @bug		No known bugs.
 */

#ifndef RENDER_H
#define RENDER_H

#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <png.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <experimental/filesystem>
//#include <essentia/algorithmfactory.h>
//#include <essentia/essentiamath.h>
//#include <essentia/pool.h>


/// group functions in namespace render
namespace render {
	/**
	 * Save a corresponding matrix to the file format PGM which stands for Portable Graymap
	 * and is part of the Netpbm project
	 * @param m the input matrix which is saved to a pgm file
	 * @return void
	 */
	void matrix_to_PGM(
			std::vector<std::vector<double>> m
			);
	
	/**
	 * Save a corresponding mfcc matrix to file where each row represents a different time stemp and
	 * the contains the different MFCC's (e.g 13 Coefficients = 13 MFC Bands)
	 * @param m the input mfcc matrix which is saved
	 * @param audioFilename the corresponding audio filenmae to save the mfcc matrix with the same name
	 * and the extension .mfcc
	 * @return void
	 */
	void matrix_to_MFCC_file(
			std::vector<std::vector<double>> m,
			std::string audioFilename
			);
	
	/**
	 * Calculates the rgb color byte on basis of the input value
	 * @param ptr a reference to the rgb byte, for storing the calculatet color
	 * @param type contains a string which describes the type <linear,log>
	 * @param val the input value which is used for the calculation of the rgb color byte
	 * @return void
	 */
	inline void set_RGB(
			png_byte *ptr,
			std::string type,
			double val
			);
	
	/**
	 * Save a vector of doubles to a png image file
	 * @param path contains the path and name of the file the png is saved under
	 * @param addStr contains an additional string like: bands, mfcc, spec
	 * @param type contains a string which describes the type <linear,log>
	 * to represent what kind of image the file contains
	 * @param height height of the image
	 * @param width width of the image
	 * @param v input double vector
	 * @return void
	 */
	void vector_to_PNG(
			std::string path,
			std::string addStr,
			std::string type,
			int unsigned height,
			int unsigned width,
			std::vector<double> v
			);
	
	void get_mfcc_from_file(std::vector<
			std::vector<double>>& mMfccCoeffs,
			std::string mfccFilename);
	
	void write_color_test_pngs();

	void vector_to_file(
			std::vector<double> v,
			std::string filename
			);
	
	void matrix_to_file(
			std::vector<std::vector<double>> m,
			std::string filename
			);
	
	void matrix_to_file(
			std::vector<std::vector<std::string>> m,
			std::string filename
			);
	
	void audio_to_matrix(
			std::vector<std::vector<double>> &m,
			std::vector<double> &audioStream
			);
}
#endif		// RENDER_H
