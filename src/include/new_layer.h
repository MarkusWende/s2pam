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

#include "new_cell.h"

class New_Layer
{
	private:
		std::vector<New_Cell> cells_;
		int id_;

	public:
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

		/**
		 * print the cells in this layer
		 * @return void
		 */
		void print_cells();

};			// end of class NEW_LAYER
#endif		// NEW_LAYER_H
