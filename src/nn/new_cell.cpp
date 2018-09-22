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
	///	assign the index to the id variable of the cell
	id_ = index;

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
		connectionsOut_.back().weight = -1;
		connectionsOut_.back().fromCell = index;
		connectionsOut_.back().toCell = i;
		connectionsOut_.back().fromLayer = layerId;
		connectionsOut_.back().toLayer = layerId + 1;
	}
}

float New_Cell::get_weight(int layerFrom, int layerTo, int cellFrom, int cellTo)
{
	float weight = -99;
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
			weight = connectionsOut_.at(i).weight;
		}
	}
	return weight;
}
		
void New_Cell::set_weights(int layerFrom, int layerTo, int cellFrom, int cellTo, float newWeight)
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
			connectionsOut_.at(i).weight = newWeight;
		}
	}
}

void New_Cell::print_connections()
{
	///	iterate the number of connections and print them
	cout << "\t\tConnections Out ----------- size: " << connectionsOut_.size() << endl;
	for (int i = 0; i < connectionsOut_.size(); i++) {
		cout << "\t\t\tOut: " << i << "\t(l: " << connectionsOut_.at(i).fromLayer << "|c: " 
			<< connectionsOut_.at(i).fromCell << ") ---> (l: " << connectionsOut_.at(i).toLayer
			<< "|c: " << connectionsOut_.at(i).toCell << ")" << "\t ---- weight: " 
			<< connectionsOut_.at(i).weight << endl;
	}
/*	cout << "\t\tConnections In ------------ size: " << connectionsIn_.size() << endl;
	for (int i = 0; i < connectionsIn_.size(); i++) {
		cout << "\t\t\tIn: " << i << "\t<--- (l: " << connectionsIn_.at(i).fromLayer
			<< "|c: " << connectionsIn_.at(i). fromCell << ")"
			<< "\t ---- weight: " << connectionsIn_.at(i).weight << endl;
	}*/
}
