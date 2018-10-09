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
	_error = 0.0;

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
				New_Cell newCell(j, 0, topo.at(i+1), i, false, 1);
				newLayer.add_cell(newCell);
				newLayer.set_type(1);
			} else if (i == (topo.size() - 1))
			{	///	output layer
				New_Cell newCell(j, topo.at(i-1), 0, i, false, 3);
				newLayer.add_cell(newCell);
				newLayer.set_type(3);
			} else 
			{	///	hidden layer
				New_Cell newCell(j, topo.at(i-1), topo.at(i+1), i, true, 2);
				newLayer.add_cell(newCell);
				newLayer.set_type(2);
			}

			//cout << "Layer: " << newLayer.get_id() << " -> Cell: " << newCell.get_id() << endl;
		}
		/**
		 * push the new filled layer to the layers_ vector
		 */
		layers_.push_back(newLayer);
	}
}

void New_Blstm::add_bias()
{
	for (int i = 0; i < layers_.size() - 1; i++) {
		vector<New_Cell> thisLayerCells;
		vector<New_Cell> nextLayerCells;
		layers_.at(i).get_cells(thisLayerCells);
		layers_.at(i+1).get_cells(nextLayerCells);
		New_Cell newCell(2, 0, nextLayerCells.size(), i, false, 4);

		for (int c = 0; c < nextLayerCells.size(); c++) {
			//newCell.create_connection(i, i+1, -1, nextLayerCells.at(c).get_id());
			if (i == 0)
				newCell.set_weight(i, i+1, 2, nextLayerCells.at(c).get_id(), 0.35);
			else
				newCell.set_weight(i, i+1, 2, nextLayerCells.at(c).get_id(), 0.6);
		}

		layers_.at(i).add_cell(newCell);
		layers_.at(i).set_bias(true);
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

			int bias = 0;
			if (layers_.at(i).get_bias())
				bias = -1;


			///	iterate over all cells in this layer
			for (int c = 0; c < thisLayerCells.size() + bias; c++) {
				
				float sum = 0.0;
				//cout << "Layer: " << i << " || Cell: " << c << endl;
				///	iterate over each cell of the previous layer
				for (int pC = 0; pC < prevLayerCells.size(); pC++) {
					float weight = prevLayerCells.at(pC).get_weight(prevLayer.get_id(), 
								layers_.at(i).get_id(),
								prevLayerCells.at(pC).get_id(),
								thisLayerCells.at(c).get_id());
					sum += weight * prevLayerCells.at(pC).get_output();
					//cout << "Weight: " << weight << "\tOutput: " << prevLayerCells.at(pC).get_output() << endl;
					//cout << "\tprev Layer id: " << prevLayer.get_id() << endl;
					//cout << "\tthis Layer id: " << layers_.at(i).get_id() << endl;
					//cout << "\tprev Layer Cell id: " << prevLayerCells.at(pC).get_id() << endl;
					//cout << "\tthis Layer Cell id: " << thisLayerCells.at(c).get_id() << endl;
				}
				/*if (i != (layers_.size() - 1) )
				{
				///	get weight of the recursive connection
				sum += thisLayerCells.at(c).get_output() * thisLayerCells.at(c).get_weight(
						layers_.at(i).get_id(), layers_.at(i).get_id(),
						thisLayerCells.at(c).get_id(), thisLayerCells.at(c).get_id());
				}*/

				///	save the sum to the input vector
				input.push_back(sum);
				//cout << "Sum: " << sum << endl;
			}
		} else
		{
			input = inVals;
		}
		/// add bias of 1
		input.push_back(1);
		
		/// pass input vector to layer feed_forward function
		layers_.at(i).feed_forward(input);
	} /// end for
}

void New_Blstm::back_prop(vector<float> targetVals)
{
	layers_.back().set_targets(targetVals);
	//layers_.back().set_error(-1);
	_error = layers_.back().get_error();

	/// iterate the different layers backwards
	for (int i = layers_.size() - 2; i >= 0; i--)
	{
		///	get the next layer
		New_Layer nextLayer = layers_.at(i+1);
		New_Layer nextButOneLayer;
		if (i < (layers_.size() - 2))
			layers_.at(i).back_prop(layers_.at(i+1), layers_.at(i+2));
		else
			layers_.at(i).back_prop(layers_.at(i+1), layers_.at(i+1));
		
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
		//layers_.at(i).weights(nextLayer);
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

float New_Blstm::get_error()
{
	float error = layers_.back().get_error();
	return error;
}

float New_Blstm::get_results()
{
	float result;
	int index = layers_.size() - 1;
	result = layers_.at(index).get_results();
}
