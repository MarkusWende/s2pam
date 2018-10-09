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

class New_Cell;

struct New_Connection
{
	float weight;
	float oldWeight;
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
		float _out;
		float _in;
		//float _error;
		//float _delta;
		float _target;
		unsigned _type;								///	1 = input layer, 2 = hidden layer, 3 = output layer

	public:
		
		float _delta;
		float _error;

		/**
		 * Constructor
		 * create a new cell
		 */
		New_Cell(
				int index,
				int numInputs,
				int numOutputs,
				int layerId,
				bool hiddenLayer,
				unsigned type
				);
		
		/**
		 * create a new connection
		 * @param layerFrom the layer the connection comes from
		 * @param layerTo the layer the connection goes to
		 * @param cellFrom the cell the connection comes from
		 * @param cellTo the cell the connection goes to
		 * @return void
		 */
		void create_connection(
				int layerFrom,
				int layerTo,
				int cellFrom,
				int cellTo);
		
		/**
		 * get the state of the cell
		 * @return the cell state as a float
		 */
		float get_Ct() { return cT_; };

		/**
		 * get the output value
		 * @return the output as a float
		 */
		float get_output() { return _out; };
		
		/**
		 * get the input value
		 * @return the input as a float
		 */
		float get_input() { return _in; };
		
		/**
		 * get the error
		 * @return the error as a float
		 */
		//float get_error() { return _error; };
		
		/**
		 * get the delta
		 * @return value as a float
		 */
		//float get_delta() const { return _delta; };
		
		/**
		 * get the target
		 * @return value as a float
		 */
		float get_target() { return _target; };
		
		/**
		 * get id of the cell
		 * @return int the id value
		 */
		int get_id() { return id_; };
		
		/**
		 * get the weight of the specified connection
		 * @param layerFrom the layer the connection comes from
		 * @param layerTo the layer the connection goes to
		 * @param cellFrom the cell the connection comes from
		 * @param cellTo the cell the connection goes to
		 * @return void
		 */
		float get_weight(
				int layerFrom,
				int layerTo,
				int cellFrom,
				int cellTo
				);

		/**
		 * get the old weight of the specified connection
		 * @param layerFrom the layer the connection comes from
		 * @param layerTo the layer the connection goes to
		 * @param cellFrom the cell the connection comes from
		 * @param cellTo the cell the connection goes to
		 * @return void
		 */
		float get_old_weight(
				int layerFrom,
				int layerTo,
				int cellFrom,
				int cellTo
				);

		/**
		 * set the cell state
		 * @param val
		 * @return void
		 */
		void set_Ct(float val) { cT_ = val; };
		
		/**
		 * set the output value
		 * @param val
		 * @return void
		 */
		void set_output(float val) { _out = val; };

		/**
		 * set the input value
		 * @param val
		 * @return void
		 */
		void set_input(float val) { _in = val; };

		/**
		 * set the error value
		 * @param val
		 * @return void
		 */
		//void set_error(float val) { _error = val; };

		/**
		 * set the delta value
		 * @param val
		 * @return void
		 */
		//void set_delta(float val) { _delta = val; };

		/**
		 * set the target value
		 * @param val
		 * @return void
		 */
		void set_target(float val) { _target = val; };

		/**
		 * set the new weight to the specified connection
		 * @param layerFrom the layer the connection comes from
		 * @param layerTo the layer the connection goes to
		 * @param cellFrom the cell the connection comes from
		 * @param cellTo the cell the connection goes to
		 * @return void
		 */
		void set_weight(
				int layerFrom,
				int layerTo,
				int cellFrom,
				int cellTo,
				float newWeight
				);

		/**
		 * print the connections of this cell
		 * @return void
		 */
		void print_connections();

};			// end of class NEW_CELL
#endif		// NEW_CELL_H
