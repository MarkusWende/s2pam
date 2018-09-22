/**
 * @file	new_blstm.h
 * @class	NEW_BLSTM
 *
 * @brief	BLSTM (Bidirectional long short term memory) Neural Network
 *
 *			This class represents the blstm neural network skeletal structure
 *
 * @note	-
 *
 * @author	Markus Wende
 * @version 1.0
 * @date	2017-2018
 * @bug		No known bugs.
 */

#ifndef NEW_BLSTM_H
#define NEW_BLSTM_H

#include <vector>
#include <iostream>

#include "new_layer.h"

class New_Blstm
{
	private:
		std::vector<New_Layer> layers_;
		std::vector<std::vector<float>> weights_;

	public:

		/**
		 * Constructor
		 * create a new blstm neural network
		 * @param topo contains the topology of the neural net e.g {4,12,12,1}
		 * => a network with an input layer with 4 cells, two hiiden layers with 12 cells each and
		 * an output layer with 1 cell
		 */
		New_Blstm(
				std::vector<unsigned> topo
				);
		
		/**
		 * feed forward function to feed the neural net with input values
		 * @param inVals contain the network input Values
		 * @return void
		 */
		void feed_forward(
				std::vector<float> inVals
				);

		/**
		 * the training of the network via back propagation
		 * @param targetVals contain the values of the ground truth
		 * @return void
		 */
		void back_prop(
				std::vector<float> targetVals
				);

		void random_weights();
		
		/**
		 * print network structure
		 * @return void
		 */
		void print_structure();
};			// end of class NEW_BLSTM
#endif		// NEW_BLSTM_H
