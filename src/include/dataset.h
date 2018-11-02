/**
 * @file	dataset.h
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

#ifndef DATASET_H
#define DATASET_H

#include <vector>		/// std::vector
#include <fstream>		/// std::ifstream
#include <iostream>		/// std::cout
#include <string>		/// std::string
#include <sstream>		/// std::stringstream
#include <algorithm>	/// std::count

class DataSet
{
	public:
		DataSet(const std::string filename);
		bool isEof() { return _file.eof(); };
		void return_to_begin_of_file();
		int size();
		void init_set(
			int T,
			std::vector<unsigned> topo,
			std::vector<std::vector<double>> &X,
			std::vector<std::vector<double>> &Y
			);
		void shift_set(
			int steps,
			std::vector<std::vector<double>> &X,
			std::vector<std::vector<double>> &Y
			);
	
	private:
		std::ifstream _file;
};			// end of class DataSet
#endif		// DATASET_H
