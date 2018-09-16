/**
 * @file	cell.h
 * @class	Cell
 *
 * @brief	Cell
 *
 *			This class represents the blstm cell structure
 *
 * @note	-
 *
 * @author	Markus Wende
 * @version 1.0
 * @date	2017-2018
 * @bug		No known bugs.
 */

#include "cell.h"

using namespace std;
double Cell::eta_ = 0.15;		// overall net learning rate, [0.0..1.0]
double Cell::alpha_ = 0.5;		// momentum, multiplier of last delatWeight, [0.0..n]

Cell::Cell(unsigned numOutputs, unsigned index)
{
	for (unsigned c = 0; c < numOutputs; ++c) {
		outputWeights_.push_back(Connection());
		outputWeights_.back().weight = random_weight_();
	}

	index_ = index;
	//cout << "Cell: " << index_ << endl;
	//cout << "Weight: " << outputWeights_.back().weight;
}

void Cell::update_input_weights(Layer &prevLayer)
{
	// The weights to be updated are in the Connection container
	// in the neurons in the preceding layer
	
	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		Cell &cell = prevLayer[n];
		double oldDeltaWeight = cell.outputWeights_[index_].deltaWeight;

		double newDeltaWeight = 
			// Individual input, magnified by the gradient and train rate:
			eta_
			* cell.get_output_val()
			* gradient_
			// Also add momentum = a faction of the previous delta weight
			+ alpha_
			* oldDeltaWeight;
		cell.outputWeights_[index_].deltaWeight = newDeltaWeight;
		cell.outputWeights_[index_].weight += newDeltaWeight;
		//cout << "OldDeltaWeight: " << oldDeltaWeight << endl;
		//cout << "NewDeltaWeight: " << newDeltaWeight << endl;
	}
}

double Cell::sum_DOW_(const Layer &nextLayer) const
{
	double sum = 0.0;

	// Sum our contributions of the errors at the nodes we feed
	
	for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
		sum += outputWeights_[n].weight * nextLayer[n].gradient_;
	}

	return sum;
}

void Cell::calc_hidden_gradients(const Layer &nextLayer)
{
	double dow = sum_DOW_(nextLayer);
	gradient_ = dow * Cell::transfer_function_derivative_(outputVal_);
}

void Cell::calc_output_gradients(double targetVal)
{
	double delta = targetVal - outputVal_;
	gradient_ = delta * Cell::transfer_function_derivative_(outputVal_);
}

double Cell::transfer_function_(double x)
{
	// tanh - output range [-1.0 .. 1.0]
	return tanh(x);
}

double Cell::transfer_function_derivative_(double x)
{
	// tanh derivative
	return 1.0 - x * x;
}

void Cell::feed_forward(const Layer &prevLayer)
{
	double sum = 0.0;

	// Sum the previous layer's outputs (which are our inputs)
	// Include the bias node from the Previous layer.
	
	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		sum += prevLayer[n].get_output_val() *
			prevLayer[n].outputWeights_[index_].weight;
		//cout << "SUM: " << sum << " || Output: " << prevLayer[n].get_output_val()
		//	<< " || Weight: " << prevLayer[n].outputWeights_[index_].weight << endl;
	}

	outputVal_ = Cell::transfer_function_(sum);
	//cout << "outputVal_: " << outputVal_ << endl;
}
