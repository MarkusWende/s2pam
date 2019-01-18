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
#include "statistic.h"

class Blstm
{
	private:
		///	weight matrix _W stores the weights of the hidden layer
		/// _b_* are the bidirectional weights
		std::vector<std::vector<double>> _Wi;
		std::vector<std::vector<double>> _Wf;
		std::vector<std::vector<double>> _Wo;
		std::vector<std::vector<double>> _Wz;
		std::vector<std::vector<double>> _b_Wi;
		std::vector<std::vector<double>> _b_Wf;
		std::vector<std::vector<double>> _b_Wo;
		std::vector<std::vector<double>> _b_Wz;

		///	weight matrix _R stores the recurrent weights of the hidden layer
		/// _b_* are the bidirectional weights
		std::vector<std::vector<double>> _Ri;
		std::vector<std::vector<double>> _Rf;
		std::vector<std::vector<double>> _Ro;
		std::vector<std::vector<double>> _Rz;
		std::vector<std::vector<double>> _b_Ri;
		std::vector<std::vector<double>> _b_Rf;
		std::vector<std::vector<double>> _b_Ro;
		std::vector<std::vector<double>> _b_Rz;
		
		/// hidden layer output weights
		std::vector<std::vector<double>> _Wy;
		std::vector<std::vector<double>> _b_Wy;

		/// bias and peepohole weights
		std::vector<double> _bi;
		std::vector<double> _bf;
		std::vector<double> _bo;
		std::vector<double> _bz;
		std::vector<double> _b_bi;
		std::vector<double> _b_bf;
		std::vector<double> _b_bo;
		std::vector<double> _b_bz;
		
		std::vector<double> _pi;
		std::vector<double> _pf;
		std::vector<double> _po;
		std::vector<double> _b_pi;
		std::vector<double> _b_pf;
		std::vector<double> _b_po;

		/// gradients
		std::vector<std::vector<double>> _gWi;
		std::vector<std::vector<double>> _gWf;
		std::vector<std::vector<double>> _gWo;
		std::vector<std::vector<double>> _gWz;
		std::vector<std::vector<double>> _b_gWi;
		std::vector<std::vector<double>> _b_gWf;
		std::vector<std::vector<double>> _b_gWo;
		std::vector<std::vector<double>> _b_gWz;

		std::vector<std::vector<double>> _gRi;
		std::vector<std::vector<double>> _gRf;
		std::vector<std::vector<double>> _gRo;
		std::vector<std::vector<double>> _gRz;
		std::vector<std::vector<double>> _b_gRi;
		std::vector<std::vector<double>> _b_gRf;
		std::vector<std::vector<double>> _b_gRo;
		std::vector<std::vector<double>> _b_gRz;
		
		std::vector<std::vector<double>> _gWy;
		std::vector<std::vector<double>> _b_gWy;

		std::vector<double> _gbi;
		std::vector<double> _gbf;
		std::vector<double> _gbo;
		std::vector<double> _gbz;
		std::vector<double> _b_gbi;
		std::vector<double> _b_gbf;
		std::vector<double> _b_gbo;
		std::vector<double> _b_gbz;
		
		std::vector<double> _gpi;
		std::vector<double> _gpf;
		std::vector<double> _gpo;
		std::vector<double> _b_gpi;
		std::vector<double> _b_gpf;
		std::vector<double> _b_gpo;

		///	hidden output states
		std::vector<std::vector<double>> _y;
		std::vector<std::vector<double>> _b_y;
		
		/// internal hidden connections
		std::vector<std::vector<double>> _f;
		std::vector<std::vector<double>> _i;
		std::vector<std::vector<double>> _o;
		std::vector<std::vector<double>> _z;
		std::vector<std::vector<double>> _b_f;
		std::vector<std::vector<double>> _b_i;
		std::vector<std::vector<double>> _b_o;
		std::vector<std::vector<double>> _b_z;
		
		std::vector<std::vector<double>> _f_head;
		std::vector<std::vector<double>> _i_head;
		std::vector<std::vector<double>> _o_head;
		std::vector<std::vector<double>> _z_head;
		std::vector<std::vector<double>> _b_f_head;
		std::vector<std::vector<double>> _b_i_head;
		std::vector<std::vector<double>> _b_o_head;
		std::vector<std::vector<double>> _b_z_head;
		
		/// hidden cell states
		std::vector<std::vector<double>> _c;
		std::vector<std::vector<double>> _b_c;

		/// output states
		std::vector<std::vector<double>> _prediction;
		std::vector<double> _predictionSingle;

		///	number of frames before and after 
		/// the target label = input and output train lenght
		int _T;
		/// learning rate determines how fast the neural network learns
		double _learningRate;
		double _momentum;
		
		/// hidden layer size
		int _hLSize;
		///	input layer size
		int _iLSize;
		///	output layer size
		int _oLSize;

	public:

		/**
		 * Constructor for existing nn filename
		 * create a new blstm neural network with the weights in the file
		 * @param filename the file containing the trained neural network
		 */
		Blstm(
				std::string filename
				);

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
		void feed_forward(
				std::vector<std::vector<double>> X
				);

		/**
		 * backward propagation function to feed the bidirectional part of
		 * the neural net with input values
		 * @param X contains the network input matrix (input train)
		 * @return void
		 */
		void feed_backward(
				std::vector<std::vector<double>> X
				);

		/**
		 * process predictions in a many-to-many network
		 * @return void
		 */
		void calculate_predictions();

		/**
		 * process predictions in a many-to-one network
		 * @return void
		 */
		void calculate_single_predictions();

		/**
		 * backpropagation through time function to train the neural network
		 * @param X is the input matrix
		 * @param Y is the ground truth or target matrix
		 * @return void
		 */
		void bptt(
				std::vector<std::vector<double>> X,
				std::vector<std::vector<double>> Y,
				std::vector<double> target
				);

		/**
		 * forwardpropagation through time function to train bidirectional part
		 * of the neural network
		 * @param X is the input matrix
		 * @param Y is the ground truth or target matrix
		 * @return void
		 */
		void fptt(
				std::vector<std::vector<double>> X,
				std::vector<std::vector<double>> Y,
				std::vector<double> target
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
		 * this function is used in a many-to-many network
		 * @param Y the target matrix
		 * @return double
		 */
		double calculate_loss(
				std::vector<std::vector<double>> Y
				);

		/**
		 * loss function to calculate the Error/Loss of the network,
		 * meaning the smaller the value the
		 * better does the network predicts the right output
		 * this function is used in a many-to-one network
		 * @param Y the target vector
		 * @return double
		 */
		double calculate_single_loss(
				std::vector<double> target
				);

		/**
		 * loss function to calculate the Error/Loss of the network
		 * with weighted labels,
		 * meaning the smaller the value the
		 * better does the network predicts the right output
		 * this function is used in a many-to-one network
		 * @param Y the target vector
		 * @return double
		 */
		double calculate_single_loss_weighted(
				std::vector<double> target
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

		/**
		 * cout the sum of the weights, to check for exploding weights
		 * @return bool, false if a certain value is reached
		 */
		bool check_weight_sum();

		/**
		 * return topology of the network
		 * @return vector<unsigned>, the topology in a unsigned vector
		 */
		std::vector<unsigned> get_topo();

		/**
		 * return the frame block size T
		 * @return int
		 */
		int get_T() { return _T; };

		/**
		 * return the predictions in a many-to-many network
		 * @return vector<vector<double>>
		 */
		std::vector<std::vector<double>> get_predictions() { return _prediction; };
		
		/**
		 * return the predictions in a many-to-one network
		 * @return vector<double>
		 */
		std::vector<double> get_single_prediction() { return _predictionSingle; };


		/**
		 * save neural network to binary file
		 * @return void
		 */
		void save();

};			// end of class BLSTM
#endif		// BLSTM_H
