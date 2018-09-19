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

New_Cell::New_Cell(int index, int numInputs, int numOutputs, int layerId)
{
	id_ = index;

	for (int i = 0; i < numInputs; i++) {
		connectionsIn_.push_back(New_Connection());
		connectionsIn_.back().weight = -1;
		connectionsIn_.back().fromCell = i;
		connectionsIn_.back().toCell = index;
		connectionsIn_.back().fromLayer = layerId - 1;
		connectionsIn_.back().toLayer = layerId;
	}
	for (int i = 0; i < numOutputs; i++) {
		connectionsOut_.push_back(New_Connection());
		connectionsOut_.back().weight = -1;
		connectionsOut_.back().fromCell = index;
		connectionsOut_.back().toCell = i;
		connectionsOut_.back().fromLayer = layerId;
		connectionsOut_.back().toLayer = layerId + 1;
	}
}
		
void New_Cell::set_weights()
{
	for (int i = 0; i < connectionsIn_.size(); i++) {
		connectionsIn_.at(i).weight = 222;
	}
	for (int i = 0; i < connectionsOut_.size(); i++) {
		connectionsOut_.at(i).weight = 333;
	}
}

void New_Cell::print_connections()
{
	cout << "\t\tConnections Out ----------- size: " << connectionsOut_.size() << endl;
	for (int i = 0; i < connectionsOut_.size(); i++) {
		cout << "\t\t\tOut: " << i << "\t---> (l: " << connectionsOut_.at(i).toLayer
			<< "|c: " << connectionsOut_.at(i).toCell << ")" << "\t ---- weight: " 
			<< connectionsOut_.at(i).weight << endl;
	}
	cout << "\t\tConnections In ------------ size: " << connectionsIn_.size() << endl;
	for (int i = 0; i < connectionsIn_.size(); i++) {
		cout << "\t\t\tIn: " << i << "\t<--- (l: " << connectionsIn_.at(i).fromLayer
			<< "|c: " << connectionsIn_.at(i). fromCell << ")"
			<< "\t ---- weight: " << connectionsIn_.at(i).weight << endl;
	}
}
