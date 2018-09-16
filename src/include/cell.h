/**
 * @file	cell.h
 * @class	Cell
 *
 * @brief	Cell
 *
 *			This class represents the blstm cell structure
 *
 * @note	-
 *
 * @author	Markus Wende
 * @version 1.0
 * @date	2017-2018
 * @bug		No known bugs.
 */

#ifndef CELL_H
#define CELL_H

#include<stdlib.h>		/// rand
#include<vector>		/// vector
#include<math.h>		/// tanh
#include<iostream>		/// cout

struct Connection
{
	double weight;
	double deltaWeight;
};

class Cell;
typedef std::vector<Cell> Layer;		

class Cell
{
	public:
		Cell(unsigned numOutputs, unsigned index);

		void set_output_val(
				double val) { outputVal_ = val; };
	
		double get_output_val(
				void) const { return outputVal_; };
		
		void feed_forward(
				const Layer &prevLayer
				);
		
		void calc_output_gradients(
				double targetVal
				);
		
		void calc_hidden_gradients(
				const Layer &nextLayer
				);
		
		void update_input_weights(
				Layer &prevLayer
				);
		
		void get_weights(
				std::vector<Connection> &weights) { weights = outputWeights_; return; };
		
		unsigned get_cell_index(
				void) { return index_; };
	
	private:
		static double eta_;		// [0.0..1.0] overall net training rate
		static double alpha_;	// [0.0..n] multiplier of last weight change (momentum)
		static double transfer_function_(double x);
		static double transfer_function_derivative_(double x);
		static double random_weight_(void) { return rand() / double(RAND_MAX); };
		double sum_DOW_(const Layer &nextLayer) const;
		double outputVal_;
		std::vector<Connection> outputWeights_;
		unsigned index_;
		double gradient_;
	
};			// end of class Cell
#endif		// CELL_H
