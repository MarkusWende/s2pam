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
	id_ = index;
}

void New_Layer::add_cell(New_Cell& newCell)
{
	//cout << newCell.get_id() << endl;
	cells_.push_back(newCell);
}

void New_Layer::set_weights()
{
	for (int i = 0; i < cells_.size(); i++) {
		cells_.at(i).set_weights();
	}
}

void New_Layer::print_cells()
{
	for (int i = 0; i < cells_.size(); i++) {
		cout << "\tCell: " << cells_.at(i).get_id() << endl;
		cells_.at(i).print_connections();
	}
}
