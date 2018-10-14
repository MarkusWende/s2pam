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

#include "helper.h"

class New_Cell;

struct New_Connection
{
	std::vector<float> weightHT;
	std::vector<float> oldWeightHT;
	float weight;
	float oldWeight;
	int fromLayer;
	int toLayer;
	int fromCell;
	int toCell;
	bool recursive;
};

class New_Cell
{
	private:
		std::vector<New_Connection> connectionsIn_;
		std::vector<New_Connection> connectionsOut_;
		int id_;
		int _T;
		float _s;									///	state of the cell C_t
		float _out;
		float _in;
		std::vector<float> _cT;
		std::vector<float> _outT;
		std::vector<float> _inT;
		std::vector<float> _inHT;
		//float _error;
		//float _delta;
		float _target;
		unsigned _type;								///	1 = input, 2 = hidden, 3 = output, 4 = bias

	public:
		
		float _delta;
		std::vector<float> _deltaT;
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
				unsigned type,
				int T
				);
		
		/**
		 * create a new connection
		 * @param layerFrom the layer the connection comes from
		 * @param layerTo the layer the connection goes to
		 * @param cellFrom the cell the connection comes from
		 * @param cellTo the cell the connection goes to
		 * @return void
		 */
		void create_recursive_connection(
				int layer,
				int cell
				);
		
		/**
		 * get the state of the cell
		 * @return the cell state as a float
		 */
		float get_s() { return _s; };

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
		 * return type of the cell
		 * @return int the id value
		 */
		unsigned get_type() { return _type; };

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
		void set_cell_T(float cT);
		
		/**
		 * set the output value
		 * @param val
		 * @return void
		 */
		void set_output(float val) { _out = val; };
		
		void set_output_T(float val);

		/**
		 * set the input value
		 * @param val
		 * @return void
		 */
		void set_input(float val) { _in = val; };

		void set_input_T(float val);
		
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

		void print_cell_T();

};			// end of class NEW_CELL
#endif		// NEW_CELL_H
