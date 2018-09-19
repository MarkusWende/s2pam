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
		 * get layer index
		 * @return int the id of the layer
		 */
		int get_id() { return id_; };

		/**
		 * get cells in layer
		 * @return vector of cells in this layer
		 */
		void get_cells(std::vector<New_Cell>& cells) { cells = cells_; };

		void add_cell(New_Cell& newCell);

		void set_weights();

		void print_cells();

};			// end of class NEW_LAYER
#endif		// NEW_LAYER_H
