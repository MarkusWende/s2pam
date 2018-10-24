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
#include <math.h>
#include <stdlib.h>		/// rand
#include <ctime>

#include "helper.h"
#include "render.h"

class Blstm
{
	private:
		///	weight matrix _U stores the weights of the input layer to the hidden layer
		std::vector<std::vector<float>> _U;
		std::vector<std::vector<float>> _Ui;
		std::vector<std::vector<float>> _Uf;
		std::vector<std::vector<float>> _Uo;
		std::vector<std::vector<float>> _Ug;
		///	weight matrix _V stores the weights of the hidden layer to the output layer
		std::vector<std::vector<float>> _V;
		///	weight matrix _W stores the weights of the hidden layer
		std::vector<std::vector<float>> _W;
		std::vector<std::vector<float>> _Wi;
		std::vector<std::vector<float>> _Wf;
		std::vector<std::vector<float>> _Wo;
		std::vector<std::vector<float>> _Wc;
		std::vector<std::vector<float>> _Wy;

		///	_dU containing the small weight changes which are applied to _U in the BPTT step
		std::vector<std::vector<float>> _dU;
		///	_dV containing the small weight changes which are applied to _V in the BPTT step
		std::vector<std::vector<float>> _dV;
		///	_dW containing the small weight changes which are applied to _W in the BPTT step
		std::vector<std::vector<float>> _dW;

		///	hidden output states
		std::vector<std::vector<float>> _s;
		std::vector<std::vector<float>> _h;
		
		/// hidden cell states
		std::vector<std::vector<float>> _c;

		/// output states
		std::vector<std::vector<float>> _o;
		std::vector<std::vector<float>> _y;

		///	number of time steps = x/y train lenght
		int _T;
		///	number of steps the bptt algrorithm is going back
		int _bpttTruncate;
		/// learning rate determines how fast the neural network learns
		float _learningRate;
		
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
				float lR
				);

		/**
		 * Activation function tanh(x)
		 * @param x the input which is activated
		 * @return float the activated value as a float
		 */
		float tanhyp(
				float x
				);

		/**
		 * Activation function tanh(x) for vector input
		 * @param x the input vector which is activated
		 * @return the activated vector value as a float vector
		 */
		std::vector<float> tanhyp(
				std::vector<float> x
				);

		/**
		 * Softmax layer function
		 * takes a float vector and squashes it in the range of [0,1]
		 * @param x input vector
		 * @return vector<float> the squashed vector
		 */
		std::vector<float> softmax(
				std::vector<float> x
				);

		/**
		 * sigmoid function for single value input
		 * takes a float and squashed into the range [0,1]
		 * @param x input value
		 * @return the squashed value
		 */
		float sigmoid(
				float x
				);

		/**
		 * sigmoid function for vector input
		 * takes a float vector and squashes it in the range of [0,1]
		 * @param x input vector
		 * @return vector<float> the squashed vector
		 */
		std::vector<float> sigmoid(
				std::vector<float> x
				);

		/**
		 * forward propagation function to feed the neural net with input values
		 * @param X contains the network input matrix (input train)
		 * @return void
		 */
		void forward_prop(
				std::vector<std::vector<float>> X
				);

		/**
		 * backpropagation through time function to train the neural network
		 * @param X is the input matrix
		 * @param Y is the ground truth or target matrix
		 * @return void
		 */
		void bptt(
				std::vector<std::vector<float>> X,
				std::vector<std::vector<float>> Y
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
		 * @retrun float the loss/error value
		 */
		float calculate_loss(
				std::vector<std::vector<float>> Y
				);

		/**
		 * A + B
		 * add two matrices with each other, Dimension of A an B has to be the same
		 * @param A input matrix A, |R^(m x n)
		 * @param B input matrix B, |R^(m x n)
		 * @return vector<vector<float>> the summarized matrix, |R^(m x n)
		 */
		std::vector<std::vector<float>> matrix_add(
				std::vector<std::vector<float>> A,
				std::vector<std::vector<float>> B
				);
		
		/**
		 * A + x * B
		 * add two matrices with each other by multiplying a constant to every
		 * element of the secound matrix,
		 * Dimension of A an B has to be the same
		 * @param A input matrix A, m x n
		 * @param B input matrix B, m x n
		 * @param x constant that B is multiplied with
		 * @return vector<vector<float>> the summarized matrix, |R^(m x n)
		 */
		std::vector<std::vector<float>> matrix_add_with_const(
				std::vector<std::vector<float>> A,
				std::vector<std::vector<float>> B,
				float x
				);
		
		/**
		 * A * B
		 * multiply two matrices with each other, column size m of A has to be
		 * the same size as row size m of B
		 * @param A input matrix A, |R^(n x m)
		 * @param B input matrix B, |R^(m x p)
		 * @return vector<vector<float>> the multiplied matrix, |R^(n x p)
		 */
		std::vector<std::vector<float>> matrix_mult(
				std::vector<std::vector<float>> A,
				std::vector<std::vector<float>> B
				);
		
		/**
		 * a * B
		 * multiply a vector with a matrix, length of a has to be
		 * the same size as column size m of B
		 * @param a input vector a, |R^(1 x m)
		 * @param B input matrix B, |R^(m x p)
		 * @return vector<float> the product, |R^p
		 */
		std::vector<float> vec_matrix_mult(
				std::vector<float> a,
				std::vector<std::vector<float>> B
				);
		
		/**
		 * add to vectors element wise together
		 * @param a input vector a, |R^m
		 * @param b input vector b, |R^m
		 * @return vector<float> output vector, |R^m
		 */
		std::vector<float> vec_ele_add(
				std::vector<float> a,
				std::vector<float> b
				);

		/**
		 * multiply to vectors element wise together
		 * @param a input vector a, |R^m
		 * @param b input vector b, |R^m
		 * @return vector<float> output vector, |R^m
		 */
		std::vector<float> vec_ele_mult(
				std::vector<float> a,
				std::vector<float> b
				);

		/**
		 * a \otimes b
		 * outer product function of two vectors
		 * @param a input vector a, |R^m
		 * @param b input vector b, |R^n
		 * @return vector<vector<float>> the outer product matrix, |R^(m x n)
		 */
		std::vector<std::vector<float>> outer(
				std::vector<float> a,
				std::vector<float> b
				);

		/**
		 * concatenate two vectors a and b with each other
		 * @param a input vector a, |R^m
		 * @param b input vector b, |R^n
		 * @return vector<float> concatenated vector, |R^(1 x (m+n))
		 */
		std::vector<float> vec_concat(
				std::vector<float> a,
				std::vector<float> b
				);

		/**
		 * print the output of the neural net by the given target matrix Y
		 * in two columns to the console
		 * @param Y target matrix Y
		 * @return void
		 */
		void print_result(
				std::vector<std::vector<float>> Y
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
