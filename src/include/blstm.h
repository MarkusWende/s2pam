/**
 * @file	blstm.h
 * @class	BLSTM
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

#ifndef BLSTM_H
#define BLSTM_H

#include <vector>
#include <iostream>
#include <cmath>
#include <stdlib.h>		/// rand
#include <ctime>
#include <chrono>		/// random seed
#include <random>		/// normal_distributuion

#include "helper.h"
#include "render.h"

class Blstm
{
	private:
		///	weight matrix _W stores the weights of the hidden layer
		std::vector<std::vector<double>> _Wi;
		std::vector<std::vector<double>> _Wf;
		std::vector<std::vector<double>> _Wo;
		std::vector<std::vector<double>> _Wz;

		std::vector<std::vector<double>> _Ri;
		std::vector<std::vector<double>> _Rf;
		std::vector<std::vector<double>> _Ro;
		std::vector<std::vector<double>> _Rz;
		
		std::vector<std::vector<double>> _Wy;

		std::vector<double> _bi;
		std::vector<double> _bf;
		std::vector<double> _bo;
		std::vector<double> _bz;
		
		std::vector<double> _pi;
		std::vector<double> _pf;
		std::vector<double> _po;

		///	hidden output states
		std::vector<std::vector<double>> _y;
		
		/// internal hidden connections
		std::vector<std::vector<double>> _f;
		std::vector<std::vector<double>> _i;
		std::vector<std::vector<double>> _o;
		std::vector<std::vector<double>> _z;
		
		std::vector<std::vector<double>> _f_head;
		std::vector<std::vector<double>> _i_head;
		std::vector<std::vector<double>> _o_head;
		std::vector<std::vector<double>> _z_head;
		
		/// hidden cell states
		std::vector<std::vector<double>> _c;

		/// output states
		std::vector<std::vector<double>> _prediction;

		///	number of time steps = x/y train lenght
		int _T;
		/// learning rate determines how fast the neural network learns
		double _learningRate;
		
		/// hidden layer size
		int _hLSize;
		///	input layer size
		int _iLSize;
		///	output layer size
		int _oLSize;

	public:

		/**
		 * Constructor
		 * create a new blstm neural network
		 * @param topo contains the topology of the neural net e.g {4,12,12,1}
		 * => a network with an input layer with 4 cells, two hiiden layers
		 * with 12 cells each and an output layer with 1 cell
		 * @param T length of the input X and output Y train
		 * @param lR learning rate
		 */
		Blstm(
				std::vector<unsigned> topo,
				int T,
				double lR
				);

		/**
		 * Activation function tanh(x)
		 * @param x the input which is activated
		 * @return double the activated value as a double
		 */
		double tanhyp(
				double x
				);

		/**
		 * derivative of the activation function tanh(x)
		 * @param x input value
		 * @return double
		 */
		double dtanhyp(
				double x
				);

		/**
		 * Activation function tanh(x) for vector input
		 * @param x the input vector which is activated
		 * @return the activated vector value as a double vector
		 */
		std::vector<double> tanhyp(
				std::vector<double> x
				);

		/**
		 * derivative of the activation function tanh(x) for vector input
		 * @param x input vector
		 * @return double vector
		 */
		std::vector<double> dtanhyp(
				std::vector<double> x
				);

		/**
		 * Softmax layer function
		 * takes a double vector and squashes it in the range of [0,1]
		 * @param x input vector
		 * @return vector<double> the squashed vector
		 */
		std::vector<double> softmax(
				std::vector<double> x
				);

		/**
		 * sigmoid function for single value input
		 * takes a double and squashed into the range [0,1]
		 * @param x input value
		 * @return the squashed value
		 */
		double sigmoid(
				double x
				);

		/**
		 * derivative of the sigmoid function for single value input
		 * @param x input value
		 * @return double
		 */
		double dsigmoid(
				double x
				);

		/**
		 * sigmoid function for vector input
		 * takes a double vector and squashes it in the range of [0,1]
		 * @param x input vector
		 * @return vector<double> the squashed vector
		 */
		std::vector<double> sigmoid(
				std::vector<double> x
				);

		/**
		 * derivative of the sigmoid function for vector input
		 * @param x input vector
		 * @return vector<double>
		 */
		std::vector<double> dsigmoid(
				std::vector<double> x
				);

		/**
		 * forward propagation function to feed the neural net with input values
		 * @param X contains the network input matrix (input train)
		 * @return void
		 */
		void forward_prop(
				std::vector<std::vector<double>> X
				);

		/**
		 * backpropagation through time function to train the neural network
		 * @param X is the input matrix
		 * @param Y is the ground truth or target matrix
		 * @return void
		 */
		void bptt(
				std::vector<std::vector<double>> X,
				std::vector<std::vector<double>> Y
				);

		/**
		 * initialize the weight matrices U, V, W with random values
		 * @return void
		 */
		void random_weights();

		/**
		 * loss function to calculate the Error/Loss of the network,
		 * meaning the smaller the value the
		 * better does the network predicts the right output
		 * @param Y the target matrix
		 * @retrun double the loss/error value
		 */
		double calculate_loss(
				std::vector<std::vector<double>> Y
				);

		/**
		 * print the output of the neural net by the given target matrix Y
		 * in two columns to the console
		 * @param Y target matrix Y
		 * @return void
		 */
		void print_result(
				std::vector<std::vector<double>> Y
				);
		
		/**
		 * render weight matrices to png files
		 * @param index append a rising index to the files end
		 * @return void
		 */
		void render_weights(
				int index
				);

		bool check_weight_sum();

		/**
		 * save neural network to binary file
		 * @return void
		 */
		void save();

		/**
		 * load neural network from binary file
		 * @return void
		 */
		void load();

};			// end of class BLSTM
#endif		// BLSTM_H
