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

#include "new_layer.h"

using namespace std;

New_Layer::New_Layer(int index)
{
	///	assign the parsed index to the layer id
	id_ = index;
}

void New_Layer::feed_forward(vector<float> inVals)
{
	for (int i = 0; i < cells_.size(); i++) {
		cells_.at(i).set_Ct(inVals.at(i));
	}
}

void New_Layer::add_cell(New_Cell& newCell)
{
	//cout << newCell.get_id() << endl;
	///	push back the new cell to the end of the layer cells_ vector
	cells_.push_back(newCell);
}

void New_Layer::random_weights(New_Layer& nextLayer)
{
	///	get the random generator a seed
	srand(time(0));

	///	initialze next layer cell vector
	vector<New_Cell> nextLayerCells;
	nextLayer.get_cells(nextLayerCells);
	
	///	iterate every cell in this layer
	for (int i = 0; i < cells_.size(); i++) {
		///	iterate every cell in the next layer
		for (int n = 0; n < nextLayerCells.size(); n++) {
			///	generate and initialze random value
			float randWeight = rand() / float(RAND_MAX);
			
			//cout << "This Layer Id: " << id_ << endl;
			//cout << "Next Layer Id: " << nextLayer.get_id() << endl;
			//cout << "This Layer Cell Id: " << cells_.at(i).get_id() << endl;
			//cout << "Next Layer Cell Id: " << nextLayerCells.at(n).get_id() << endl;
			//cout << "New Weight: " << randWeight << endl;
			
			///	assign the random value to the specified connection
			cells_.at(i).set_weights(id_, nextLayer.get_id(), cells_.at(i).get_id(),
					nextLayerCells.at(n).get_id(), randWeight);
		}
	}
}

void New_Layer::print_cells()
{
	///	iterate every cell in the layer
	for (int i = 0; i < cells_.size(); i++) {
		cout << "\tCell: " << cells_.at(i).get_id() << endl;
		cells_.at(i).print_connections();
	}
}
