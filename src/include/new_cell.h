/**
 * @file	new_cell.h
 * @class	NEW_CELL
 *
 * @brief	Cells are the neurons of the neural net
 *
 *			This class represents the cell structure
 *
 * @note	-
 *
 * @author	Markus Wende
 * @version 1.0
 * @date	2017-2018
 * @bug		No known bugs.
 */

#ifndef NEW_CELL_H
#define NEW_CELL_H

#include <vector>
#include <string>
#include <iostream>
#include <stdlib.h>		/// rand

class New_Cell;

struct New_Connection
{
	float weight;
	int fromLayer;
	int toLayer;
	int fromCell;
	int toCell;

};

class New_Cell
{
	private:
		std::vector<New_Connection> connectionsIn_;
		std::vector<New_Connection> connectionsOut_;
		int id_;
		float cT_;									///	state of the cell C_t

	public:
		
		/**
		 * Constructor
		 * create a new cell
		 */
		New_Cell(
				int index,
				int numInputs,
				int numOutputs,
				int layerId
				);

		/**
		 * get the state of the cell
		 * @return the cell state as an float
		 */
		float get_Ct() { return cT_; };

		/**
		 * get id of the cell
		 * @return int the id value
		 */
		int get_id() { return id_; };
		
		/**
		 * set the cell state
		 * @param val
		 * @return void
		 */
		void set_Ct(float val) { cT_ = val; };

		void set_weights();

		void print_connections();

};			// end of class NEW_CELL
#endif		// NEW_CELL_H
