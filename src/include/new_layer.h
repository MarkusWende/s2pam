/**
 * @file	new_layer.h
 * @class	NEW_LAYER
 *
 * @brief	Layers contain the neurons of a neural net
 *
 *			This class represents the layers of the neural net
 *
 * @note	-
 *
 * @author	Markus Wende
 * @version 1.0
 * @date	2017-2018
 * @bug		No known bugs.
 */

#ifndef NEW_LAYER_H
#define NEW_LAYER_H

#include <iostream>
#include <stdlib.h>		/// rand
#include <ctime>
#include <math.h>		///	exp()

#include "new_cell.h"

class New_Layer
{
	private:
		std::vector<New_Cell> cells_;
		int id_;
		unsigned _type;
		float _error;
		bool _bias;
		bool _recursive;

	public:
		/**
		 * Empty Constructor
		 * create an empty new layer
		 */
		New_Layer() {};

		/**
		 * Constructor
		 * create a new layer
		 * @param numCells number of the cells the layer contains
		 */
		New_Layer(int index);

		/**
		 * feed forward function to feed the layer with the new input values
		 * @param inVals contain the network input Values
		 * @return void
		 */
		void feed_forward(
				std::vector<float> inVals
				);

		void forward_prop(
				std::vector<float> inVals,
				std::vector<float> inHVals
				);
		
		void back_prop(
				New_Layer& nextLayer,
				New_Layer& nextButOneLayer
				);
		
		float sigmoid(float x);
		float tanhyp(float x);
		float sigmoid_derivative(float out);
		float tanhyp_derivative(float x);
		
		bool is_recursive() { return _recursive; };
		
		void get_target_vals(std::vector<float>& targets);
		
		/**
		 * get layer index
		 * @return int the id of the layer
		 */
		int get_id() { return id_; };

		/**
		 * get cells in layer
		 * @return vector of cells in this layer
		 */
		void get_cells(std::vector<New_Cell>& cells) { cells = cells_; };

		std::vector<New_Cell>& get_cell_vector() { return cells_; };

		/**
		 * get layer type
		 * @return the layer type
		 */
		unsigned get_type() { return _type; };

		float get_error();

		float get_results();

		bool get_bias() { return _bias; };

		/**
		 * set layer type
		 * @param val contain the layer type
		 * @return void
		 */
		void set_type(unsigned val) { _type = val; };

		/**
		 * set target vals
		 * @param tarVals contain the network target vals
		 * @return void
		 */
		void set_targets(std::vector<float> tarVals);

		void create_recursive_connection(int layer);

		void set_error(float error);

		void set_bias(bool val) { _bias = val; };
		
		void set_recursion(bool val) { _recursive = val; };
		
		/**
		 * add a new cell to the layer
		 * @param newCell the cell that will be added
		 * @return void
		 */
		void add_cell(New_Cell& newCell);

		/**
		 * assign to every connection in this layer random values
		 * @param nextLayer parse the next Layer to the function to get the next Layer Cells
		 * @return void
		 */
		void random_weights(New_Layer& nextLayer);

		void weights(New_Layer& nextLayer);

		/**
		 * print the cells in this layer
		 * @return void
		 */
		void print_cells();

};			// end of class NEW_LAYER
#endif		// NEW_LAYER_H
