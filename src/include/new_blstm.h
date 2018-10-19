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
#include <math.h>
#include <stdlib.h>		/// rand
#include <ctime>

#include "new_layer.h"

class New_Blstm
{
	private:
		std::vector<New_Layer> layers_;
		std::vector<std::vector<float>> _U;
		std::vector<std::vector<float>> _V;
		std::vector<std::vector<float>> _W;

		std::vector<std::vector<float>> _dU;
		std::vector<std::vector<float>> _dV;
		std::vector<std::vector<float>> _dW;

		std::vector<std::vector<float>> _s;
		std::vector<std::vector<float>> _o;

		float _error;
		
		///	number of time steps
		int _T;
		int _hLSize;
		int _iLSize;
		int _oLSize;
		int _bpttTruncate;
		float _learningRate;

	public:

		/**
		 * Constructor
		 * create a new blstm neural network
		 * @param topo contains the topology of the neural net e.g {4,12,12,1}
		 * => a network with an input layer with 4 cells, two hiiden layers with 12 cells each and
		 * an output layer with 1 cell
		 */
		New_Blstm(
				std::vector<unsigned> topo,
				int T,
				float lR
				);

		void add_bias();

		void add_recursion();
		
		/**
		 * feed forward function to feed the neural net with input values
		 * @param inVals contain the network input Values
		 * @return void
		 */
		void feed_forward(
				std::vector<float> inVals
				);

		float tanhyp(float x);

		std::vector<float> tanhyp_vec(
				std::vector<float> x
				);

		std::vector<std::vector<float>> outer(
				std::vector<float> a,
				std::vector<float> b
				);

		std::vector<float> softmax(std::vector<float> x);

		/**
		 * feed forward function to feed the neural net with input values
		 * @param inVals contain the network input Values
		 * @return void
		 */
		void forward_prop(
				std::vector<std::vector<float>> X
				);

		/**
		 * the training of the network via back propagation
		 * @param targetVals contain the values of the ground truth
		 * @return void
		 */
		void back_prop(
				std::vector<float> targetVals
				);

		void bptt(
				std::vector<std::vector<float>> X,
				std::vector<std::vector<float>> Y
				);

		void random_weights();

		float calculate_loss(
				std::vector<std::vector<float>> Y
				);

		std::vector<std::vector<float>> matrix_add(
				std::vector<std::vector<float>> A,
				std::vector<std::vector<float>> B
				);
		
		std::vector<std::vector<float>> matrix_add_with_const(
				std::vector<std::vector<float>> A,
				std::vector<std::vector<float>> B,
				float x
				);
		
		std::vector<std::vector<float>> matrix_mult(
				std::vector<std::vector<float>> A,
				std::vector<std::vector<float>> B
				);
		
		/**
		 * print network structure
		 * @return void
		 */
		void print_structure();

		void print_result(std::vector<std::vector<float>> Y);

		/**
		 * get the total error
		 * @return float
		 */
		float get_error();

		float get_results();

		void set_target_vals(std::vector<float> targets);
};			// end of class NEW_BLSTM
#endif		// NEW_BLSTM_H
