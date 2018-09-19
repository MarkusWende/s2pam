/**
 * @file	new_blstm.cpp
 * @class	NEW_BLSTM
 *
 * @brief	BLSTM (Bidirectional long short term memory) Neural Network
 *
 *			This class represents the blstm neural network skeletal structure
 *
 * @note	-
 *
 * @author	Markus Wende
 * @version 1.0
 * @date	2017-2018
 * @bug		No known bugs.
 */

#include "new_cell.h"
#include "new_blstm.h"

using namespace std;

New_Blstm::New_Blstm(vector<unsigned> topo)
{
	///	loop over the different layers (column) in topo
	for (int i = 0; i < topo.size(); i++)
	{
		/**
		 * create a new layer with the index i
		 */
		New_Layer newLayer(i);

		///	loop over the values of the columns in topo => values = number of cells
		for (int j = 0; j < topo.at(i); j++)
		{
			/**
			 * create a new cell with the id j and add that cell to the actual layer
			 */
			if (i == 0)
			{
				New_Cell newCell(j, 0, topo.at(i+1), i);
				newLayer.add_cell(newCell);
			} else if (i == (topo.size() - 1))
			{
				New_Cell newCell(j, topo.at(i-1), 0, i);
				newLayer.add_cell(newCell);
			} else 
			{
				New_Cell newCell(j, topo.at(i-1), topo.at(i+1), i);
				newLayer.add_cell(newCell);
			}

			//cout << "Layer: " << newLayer.get_id() << " -> Cell: " << newCell.get_id() << endl;
		}
		/**
		 * push the new filled layer to the layers_ vector
		 */
		layers_.push_back(newLayer);
	}
}

void New_Blstm::random_weights()
{
	for (int i = 0; i < layers_.size(); i++) {
		layers_.at(i).set_weights();
	}
}

void New_Blstm::print_structure()
{
	for (int i = 0; i < layers_.size(); i++) {
		cout << "Layer: " << layers_.at(i).get_id() << endl;
		layers_.at(i).print_cells();
	}
}
