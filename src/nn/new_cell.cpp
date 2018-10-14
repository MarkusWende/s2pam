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

#include "new_cell.h"

using namespace std;

New_Cell::New_Cell(int index, int numInputs, int numOutputs, int layerId, bool hiddenLayer, unsigned type, int T)
{
	///	assign the index to the id variable of the cell
	id_ = index;
	_s = 0.0;
	_out = 0.0;
	_in = 0.0;
	_error = 0.0;
	_delta	= 99.0;
	_target = 0.0;
	_type = type;
	_T = T;
	_cT.assign(T, 0);
	_inT.assign(T, 0);
	_inHT.assign(T, 0);
	_outT.assign(T, 0);
	_deltaT.assign(T, 0);
		
/*	for (int i = 0; i < numInputs; i++) {
		connectionsIn_.push_back(New_Connection());
		connectionsIn_.back().weight = -1;
		connectionsIn_.back().fromCell = i;
		connectionsIn_.back().toCell = index;
		connectionsIn_.back().fromLayer = layerId - 1;
		connectionsIn_.back().toLayer = layerId;
	}*/
	///	iterate the number of output connections of this cell and assign the corresponding values
	for (int i = 0; i < numOutputs; i++) {
		connectionsOut_.push_back(New_Connection());
		connectionsOut_.back().weight = 0;
		connectionsOut_.back().oldWeight = 0;
		connectionsOut_.back().fromCell = index;
		connectionsOut_.back().toCell = i;
		connectionsOut_.back().fromLayer = layerId;
		connectionsOut_.back().toLayer = layerId + 1;
		connectionsOut_.back().recursive = false;
		connectionsOut_.back().weightHT.assign(T, 0);
	}
	/*		
	///	add recursive connection to cell
	if (hiddenLayer)
	{
		connectionsOut_.push_back(New_Connection());
		connectionsOut_.back().weight = 0;
		connectionsOut_.back().oldWeight = 0;
		connectionsOut_.back().fromCell = index;
		connectionsOut_.back().toCell = index;
		connectionsOut_.back().fromLayer = layerId;
		connectionsOut_.back().toLayer = layerId;
	}*/
}

void New_Cell::print_cell_T()
{
	cout << "Cell: " << id_ << "\tType: " << _type << endl;
	helper::print_vector("_cT: ", _cT);
	helper::print_vector("_inT: ", _inT);
	helper::print_vector("_inHT: ", _inHT);
	helper::print_vector("_outT: ", _outT);
	helper::print_vector("_deltaT: ", _deltaT);

	cout << "Connection Size: " << connectionsOut_.size() << endl;
	for (int i = 0; i < connectionsOut_.size(); i++) {
		helper::print_vector("weightT: ", connectionsOut_.at(i).weightHT);
	}
}

void New_Cell::print_connections()
{
	///	iterate the number of connections and print them
	cout << "\t\tConnections Out ----------- size: " << connectionsOut_.size() << endl;
	for (int i = 0; i < connectionsOut_.size(); i++) {
		if (connectionsOut_.at(i).recursive)
		{
			cout << "\t\t\tRecursive: " << i << "\t(l: " << connectionsOut_.at(i).fromLayer << "|c: " 
				<< connectionsOut_.at(i).fromCell << ") ---> (l: " << connectionsOut_.at(i).toLayer
				<< "|c: " << connectionsOut_.at(i).toCell << ")" << "\t ---- weight: " 
				<< connectionsOut_.at(i).weight << endl;
		} else
		{
			cout << "\t\t\tOut: " << i << "\t(l: " << connectionsOut_.at(i).fromLayer << "|c: " 
				<< connectionsOut_.at(i).fromCell << ") ---> (l: " << connectionsOut_.at(i).toLayer
				<< "|c: " << connectionsOut_.at(i).toCell << ")" << "\t\t ---- weight: " 
				<< connectionsOut_.at(i).weight << endl;
		}
	}
/*	cout << "\t\tConnections In ------------ size: " << connectionsIn_.size() << endl;
	for (int i = 0; i < connectionsIn_.size(); i++) {
		cout << "\t\t\tIn: " << i << "\t<--- (l: " << connectionsIn_.at(i).fromLayer
			<< "|c: " << connectionsIn_.at(i). fromCell << ")"
			<< "\t ---- weight: " << connectionsIn_.at(i).weight << endl;
	}*/
}

void New_Cell::create_recursive_connection(int layer, int cell)
{
	connectionsOut_.push_back(New_Connection());
	connectionsOut_.back().weight = 9999;
	connectionsOut_.back().oldWeight = 9999;
	connectionsOut_.back().fromCell = cell;
	connectionsOut_.back().toCell = cell;
	connectionsOut_.back().fromLayer = layer;
	connectionsOut_.back().toLayer = layer;
	connectionsOut_.back().recursive = true;
	connectionsOut_.back().weightHT.assign(_T, 0);
}

float New_Cell::get_weight(int layerFrom, int layerTo, int cellFrom, int cellTo)
{
	///	iterate the number of connections and return the new weight if the dependencies match
	//cout << "-- Connection Size: " << connectionsOut_.size() << endl;
	for (int i = 0; i < connectionsOut_.size(); i++) {
	/*	cout << "Layer From:\t" << layerFrom << " -- " << connectionsOut_.at(i).fromLayer << endl;
		cout << "Layer To:\t" << layerTo << " -- " << connectionsOut_.at(i).toLayer << endl;
		cout << "Cell From:\t" << cellFrom << " -- " << connectionsOut_.at(i).fromCell << endl;
		cout << "Cell To:\t" << cellTo << " -- " << connectionsOut_.at(i).toCell << endl;
	*/	
		if (layerFrom == connectionsOut_.at(i).fromLayer 
				&& layerFrom == connectionsOut_.at(i).fromLayer 
				&& layerTo == connectionsOut_.at(i).toLayer
				&& cellFrom == connectionsOut_.at(i).fromCell 
				&& cellTo == connectionsOut_.at(i).toCell)
		{
			return connectionsOut_.at(i).weight;
		}
	}
	return 0;
}

float New_Cell::get_old_weight(int layerFrom, int layerTo, int cellFrom, int cellTo)
{
	///	iterate the number of connections and return the new weight if the dependencies match
	//cout << "-- Connection Size: " << connectionsOut_.size() << endl;
	for (int i = 0; i < connectionsOut_.size(); i++) {
	/*	cout << "Layer From:\t" << layerFrom << " -- " << connectionsOut_.at(i).fromLayer << endl;
		cout << "Layer To:\t" << layerTo << " -- " << connectionsOut_.at(i).toLayer << endl;
		cout << "Cell From:\t" << cellFrom << " -- " << connectionsOut_.at(i).fromCell << endl;
		cout << "Cell To:\t" << cellTo << " -- " << connectionsOut_.at(i).toCell << endl;
	*/	
		if (layerFrom == connectionsOut_.at(i).fromLayer 
				&& layerFrom == connectionsOut_.at(i).fromLayer 
				&& layerTo == connectionsOut_.at(i).toLayer
				&& cellFrom == connectionsOut_.at(i).fromCell 
				&& cellTo == connectionsOut_.at(i).toCell)
		{
			return connectionsOut_.at(i).oldWeight;
		}
	}
	return 0;
}

void New_Cell::set_output_T(float out)
{
	for (int i = _outT.size() - 1; i >= 0; i--) {
		float tmp = _outT.at(i);
		_outT.at(i) = out;
		out = tmp;
	}
}

void New_Cell::set_cell_T(float cT)
{
	for (int i = _cT.size() - 1; i >= 0; i--) {
		float tmp = _cT.at(i);
		_cT.at(i) = cT;
		cT = tmp;
	}
}

void New_Cell::set_input_T(float in)
{
	for (int i = _inT.size() - 1; i >= 0; i--) {
		float tmp = _inT.at(i);
		_inT.at(i) = in;
		in = tmp;
	}
}

void New_Cell::set_weight(int layerFrom, int layerTo, int cellFrom, int cellTo, float newWeight)
{
	/*for (int i = 0; i < connectionsIn_.size(); i++) {
		if (layerFrom == connectionsIn_.at(i).fromLayer 
				&& layerFrom == connectionsIn_.at(i).fromLayer 
				&& layerTo == connectionsIn_.at(i).toLayer
				&& cellFrom == connectionsIn_.at(i).fromCell 
				&& cellTo == connectionsIn_.at(i).toCell)
		{
			connectionsIn_.at(i).weight = newWeight;
		}
	}*/
	///	iterate the number of connections and assign the new weight if the dependencies match
	for (int i = 0; i < connectionsOut_.size(); i++) {
		if (layerFrom == connectionsOut_.at(i).fromLayer 
				&& layerFrom == connectionsOut_.at(i).fromLayer 
				&& layerTo == connectionsOut_.at(i).toLayer
				&& cellFrom == connectionsOut_.at(i).fromCell 
				&& cellTo == connectionsOut_.at(i).toCell)
		{
			float tmp = connectionsOut_.at(i).weight;
			connectionsOut_.at(i).oldWeight = tmp;
			connectionsOut_.at(i).weight = newWeight;
		}
	}
}
