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
	///	iterate the number of different layers (number of columns) in topo
	for (int i = 0; i < topo.size(); i++)
	{
		/**
		 * create a new layer with the index i
		 */
		New_Layer newLayer(i);

		///	iterate the size of the values of the columns in topo => values = number of cells
		for (int j = 0; j < topo.at(i); j++)
		{
			/**
			 * create a new cell with the id j and add that cell to the actual layer
			 */
			if (i == 0)
			{	///	input layer
				New_Cell newCell(j, 0, topo.at(i+1), i, false);
				newLayer.add_cell(newCell);
			} else if (i == (topo.size() - 1))
			{	///	output layer
				New_Cell newCell(j, topo.at(i-1), 0, i, false);
				newLayer.add_cell(newCell);
			} else 
			{	///	hidden layer
				New_Cell newCell(j, topo.at(i-1), topo.at(i+1), i, true);
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

void New_Blstm::feed_forward(vector<float> inVals)
{
	///	input vector
	vector<float> input;
	/// iterate the different layers
	for (int i = 0; i < layers_.size(); i++) {
	
		/// clear input vector
		input.clear();

		/// calculate values when its not the input layer
		if (i > 0)
		{
			///	get the prev layer
			New_Layer& prevLayer = layers_.at(i-1);
			
			///	initialze previous layer cell vector
			vector<New_Cell> prevLayerCells;
			vector<New_Cell> thisLayerCells;
			layers_.at(i-1).get_cells(prevLayerCells);
			layers_.at(i).get_cells(thisLayerCells);

			///	iterate over all cells in this layer
			for (int c = 0; c < thisLayerCells.size(); c++) {
				
				float sum = 0.0;
				//cout << "Layer: " << i << " || Cell: " << c << endl;
				///	iterate over each cell of the previous layer
				for (int pC = 0; pC < prevLayerCells.size(); pC++) {
					float weight = prevLayerCells.at(pC).get_weight(prevLayer.get_id(), 
								layers_.at(i).get_id(),
								prevLayerCells.at(pC).get_id(),
								thisLayerCells.at(c).get_id());
					if (weight != -1)
					{
						sum += weight * prevLayerCells.at(pC).get_Ct();
						//cout << "Weight Sum: " << sum << endl;
						//cout << "\tprev Layer id: " << prevLayer.get_id() << endl;
						//cout << "\tthis Layer id: " << layers_.at(i).get_id() << endl;
						//cout << "\tprev Layer Cell id: " << prevLayerCells.at(pC).get_id() << endl;
						//cout << "\tthis Layer Cell id: " << thisLayerCells.at(c).get_id() << endl;
					}
				}
				if (i != (layers_.size() - 1) )
				///	get weight of the recursive connection
				sum += thisLayerCells.at(c).get_Ct() * thisLayerCells.at(c).get_weight(
						layers_.at(i).get_id(), layers_.at(i).get_id(),
						thisLayerCells.at(c).get_id(), thisLayerCells.at(c).get_id());

				///	save sum to inVals vector
				input.push_back(sum);
			}
		} else
		{
			input = inVals;
		}
		/// pass input vector to layer feed_forward function
		layers_.at(i).feed_forward(input);
	} /// end for

}

void New_Blstm::back_prop(vector<float> targetVals)
{
	///	target values vector
	vector<float> target;
	/// iterate the different layers backwards
	for (int i = layers_.size() - 1; i >= 0; i--)
	{
		///	clear target vector
		target.clear();

		if (i > 0)
		{
			///	get the prev layer
			New_Layer& prevLayer = layers_.at(i-1);
			
			///	initialze previous layer cell vector
			vector<New_Cell> prevLayerCells;
			vector<New_Cell> thisLayerCells;
			layers_.at(i-1).get_cells(prevLayerCells);
			layers_.at(i).get_cells(thisLayerCells);

			for (int c = 0; c < thisLayerCells.size(); c++)
			{
				cout << "Error: " << targetVals.at(0) - thisLayerCells.at(c).get_Ct()  << endl;
			}
		}
	}

}

void New_Blstm::random_weights()
{
	/// iterate the different layers - 1
	/// (we dont need the last layer, because there are no output connections)
	for (int i = 0; i < layers_.size() - 1; i++) {
		///	get the next layer
		/// and pass it to the random_weights() function of this layer
		New_Layer nextLayer = layers_.at(i+1);
		layers_.at(i).random_weights(nextLayer);
	}
}

void New_Blstm::print_structure()
{
	///	iterate every cell in this layer, print the id and call the cell print function
	for (int i = 0; i < layers_.size(); i++) {
		cout << "Layer: " << layers_.at(i).get_id() << endl;
		layers_.at(i).print_cells();
	}
}
