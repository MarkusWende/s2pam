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
	_error = 0.0;
	_bias = false;
}

void New_Layer::feed_forward(vector<float> inVals)
{
	for (int i = 0; i < cells_.size(); i++) {
		float out;
		if (_type == 1)
			out	= inVals.at(i);
		else if (_type == 3)
		{
			cout << "Output: ";
			out	= sigmoid(inVals.at(i));
			cout << out << "\n";
		}
		else if ( i == cells_.size())
			out	= inVals.at(i);
		else
			out	= sigmoid(inVals.at(i));

		cells_.at(i).set_input(inVals.at(i));
		cells_.at(i).set_output(out);

	}
	//cout << endl;
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
			cells_.at(i).set_weight(id_, nextLayer.get_id(), cells_.at(i).get_id(),
					nextLayerCells.at(n).get_id(), randWeight);
		}
		///	set random weight to the recursive connection
		float randWeight = rand() / float(RAND_MAX);
		cells_.at(i).set_weight(id_, id_, cells_.at(i).get_id(),
				cells_.at(i).get_id(), randWeight);
	}
}

void New_Layer::weights(New_Layer& nextLayer)
{
	///	get the random generator a seed
	srand(time(0));

	///	initialze next layer cell vector
	vector<New_Cell> nextLayerCells;
	nextLayer.get_cells(nextLayerCells);

	vector<float> weights;
	if (nextLayer.get_id() == 2)
		weights = {0.4, 0.45, 0.5, 0.55};
	if (nextLayer.get_id() == 1)
		weights = {0.15, 0.2, 0.25, 0.3};
	
	///	iterate every cell in this layer
	for (int i = 0; i < nextLayerCells.size(); i++) {
		///	iterate every cell in the next layer
		for (int n = 0; n < cells_.size(); n++) {
			///	assign the random value to the specified connection
			cells_.at(n).set_weight(id_, nextLayer.get_id(), cells_.at(n).get_id(),
					nextLayerCells.at(i).get_id(), weights.at(i*2 + n*1));
		}
		///	set random weight to the recursive connection
		//cells_.at(n).set_weight(id_, id_, cells_.at(n).get_id(),
		//		cells_.at(i).get_id(), 0);
	}
}

void New_Layer::back_prop(New_Layer& nextLayer, New_Layer& nextButOneLayer)
{
	///	set the learning rate eta
	float eta = 0.5;

	int bias = 0;
	int biasNextLayer = 0;
	int biasNextButOneLayer = 0;

	if (_bias)
		bias = -1;
	if (nextLayer.get_bias())
		biasNextLayer = -1;
	if (nextButOneLayer.get_bias())
		biasNextButOneLayer = -1;

	/// loop all cells in this layer	
	for (int c = 0; c < cells_.size() + bias; c++)
	{
		vector<New_Cell>& nextLayerCells = nextLayer.get_cell_vector();
		//nextLayer.get_cells(nextLayerCells);


		for (int nC = 0; nC < nextLayerCells.size() + biasNextLayer; nC++)
		{

			float newWeight, oldWeight, gradient;

			oldWeight = cells_.at(c).get_weight(id_, nextLayer.get_id(), cells_.at(c).get_id(),
					nextLayerCells.at(nC).get_id());

			unsigned nextLayerType = nextLayer.get_type();

			///	gradient = dE_total / dw_i
			gradient = 0.0;





			///	if cell type is an output
			if (nextLayer.get_type() == 3)
			{
				float delta = 0.0;
				float out = nextLayerCells.at(nC).get_output();
				float target = nextLayerCells.at(nC).get_target();

				float error = (out - target);
				float error2 = (target - out) * (target - out) / 2;
				nextLayerCells.at(nC)._error = error2;
				//cout << "Error: " << error << "\tOut: " << out << "\tTarget: " << target << endl;
				delta = error * out * (1 - out);

				//cout << "DEEELLLTTAA: " << nextLayerCells.at(nC)._delta << endl;

				nextLayerCells.at(nC)._delta = delta;

				//cout << "Address: " << &nextLayerCells.at(nC) << endl;

				gradient = delta * cells_.at(c).get_output();
				//cout << "Gradient:" << endl;
				//cout << gradient << " = " << delta << " * " << cells_.at(c).get_output() << endl;

			///	if cell type is a hidden
			} else if (nextLayer.get_type() == 2)
			{
				vector<New_Cell> nextButOneLayerCells;
				nextButOneLayer.get_cells(nextButOneLayerCells);
				

				float delta = 0.0;
				for (int nBOC = 0; nBOC < nextButOneLayerCells.size() + biasNextButOneLayer; nBOC++) {
					float tmpDelta = nextButOneLayerCells.at(nBOC)._delta;
					oldWeight = nextLayerCells.at(nC).get_old_weight(nextLayer.get_id(), nextButOneLayer.get_id(),
							nextLayerCells.at(nC).get_id(), nextButOneLayerCells.at(nBOC).get_id());
					//cout << "-> Address: " << &nextButOneLayerCells.at(nBOC) << endl;
					//cout << "old Weight: " << oldWeight << "\ttmp delta: " << tmpDelta << endl;
					delta += tmpDelta * oldWeight;
				}


				nextLayerCells.at(nC)._delta = delta;

				float out = nextLayerCells.at(nC).get_output();
				float in; 
				
				if (_type == 1)
					in = cells_.at(c).get_input();
				else
					in = cells_.at(c).get_output();

				gradient = delta * out * (1 - out) * in;
			}

			/// get current weight of the current layer cell to the next layer cell
			oldWeight = cells_.at(c).get_weight(id_, nextLayer.get_id(),
					cells_.at(c).get_id(), nextLayerCells.at(nC).get_id());


			newWeight = oldWeight - ( eta * gradient );
			//cout << "new Weight:" << endl;
			//cout << newWeight << " = " << oldWeight << " - " << eta << " * " << gradient << endl << endl;

			///	assign the new weight to the specified connection
			cells_.at(c).set_weight(id_, nextLayer.get_id(), cells_.at(c).get_id(),
					nextLayerCells.at(nC).get_id(), newWeight);
		}
	}
}

float New_Layer::sigmoid(float val)
{
	float sig = 0.0;
	sig = 1 / (1 + exp(-val));

	return sig;
}

void New_Layer::set_targets(vector<float> tarVals)
{
	int i = 0;
	for (int c = 0; c < cells_.size(); c++) {
		float target = tarVals.at(i);
		
		cells_.at(c).set_target(target);
		i++;
	}
}

float New_Layer::get_error()
{
	float error = 0;
	for (int c = 0; c < cells_.size(); c++) {
		error += cells_.at(c)._error;
	}
	return error;
}

float New_Layer::get_results()
{
	float result;
	result = cells_.at(0).get_output();
	return result;
}

float New_Layer::sigmoid_derivative(float val)
{
	float div = 0.0;
	div = (exp(-val)) / ( (1 + exp(-val)) * (1 + exp(-val)) );

	return div;
}

void New_Layer::get_target_vals(vector<float>& targets)
{
	targets.clear();
	for (int c = 0; c < cells_.size(); c++) {
		float tmp;
		tmp = cells_.at(c).get_target();
		targets.push_back(tmp);
	}
}

void New_Layer::print_cells()
{
	///	iterate every cell in the layer
	for (int i = 0; i < cells_.size(); i++) {
		cout << "\tCell: " << cells_.at(i).get_id() << "\tIn: " << cells_.at(i).get_input()
			<< "\tOut: " << cells_.at(i).get_output() << "\tTarget: " << cells_.at(i).get_target() << endl;
		cells_.at(i).print_connections();
	}
}
