/**
 * @file		blstm.cpp
 * @class		BLSTM
 *
 * @brief		BLSTM (Bidirectional long short term memory) Neural Network
 *
 *				This class represents the blstm cell structure
 *
 * @note		-
 *
 * @author	Markus Wende
 * @version 1.0
 * @date	2017-2018
 * @bug		No known bugs.
 */

#include "blstm.h"

using namespace std;

Blstm::Blstm(const vector<unsigned> &topology)
{
	unsigned numLayers = topology.size();
	for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
		layers_.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() -1 ? 0 : topology[layerNum + 1];

		// We have made a new Layer, now fill it with cells and
		// add a bias neuron to the layer
		for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
			layers_.back().push_back(Cell(numOutputs, neuronNum));
		}

		// Force the bias node's output value to 1.0. It's the last neuron created above
		layers_.back().back().set_output_val(1.0);
	}
}

void Blstm::back_prop(const vector<double> &targetVals)
{
	// Calculate overall net error (RMS of output neuron errors)
	
	Layer &outputLayer = layers_.back();
	error_ = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
		double delta = targetVals[n] - outputLayer[n].get_output_val();
		error_ += delta * delta;
	}
	error_ /= outputLayer.size() - 1;
	error_ = sqrt(error_); // RMS

	// Implement a recent average measurement:

	recentAverageError_ =
		(recentAverageError_ * recentAverageSmoothingFactor_ + error_)
		/ (recentAverageSmoothingFactor_ + 1.0);

	//cout << recentAverageSmoothingFactor_ << endl;
	// Calculate output layer gradients
	
	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
		outputLayer[n].calc_output_gradients(targetVals[n]);
	}

	// Calculate gradients on hidden layers
	
	for (unsigned layerNum = layers_.size() - 2; layerNum > 0; --layerNum) {
		Layer &hiddenLayer = layers_[layerNum];
		Layer &nextLayer = layers_[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
			hiddenLayer[n].calc_hidden_gradients(nextLayer);
		}
	}

	// For all layers from output to first hidden layer,
	// update connection weights
	
	for (unsigned layerNum = layers_.size() - 1; layerNum > 0; --layerNum) {
		Layer &layer = layers_[layerNum];
		Layer &prevLayer = layers_[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n) {
			layer[n].update_input_weights(prevLayer);
		}
	}
}

void Blstm::feed_forward(const vector<double> &inputVals)
{
	assert(inputVals.size() == layers_[0].size() - 1);

	// Assign (latch) the input values into the input neurons
	for (unsigned i = 0; i < inputVals.size(); ++i) {
		layers_[0][i].set_output_val(inputVals[i]);
	}

	// Forward propagate
	for (unsigned layerNum = 1; layerNum < layers_.size(); ++layerNum) {
		Layer &prevLayer = layers_[layerNum - 1];
		for (unsigned n = 0; n < layers_[layerNum].size() - 1; ++n) {
			layers_[layerNum][n].feed_forward(prevLayer);
		}
	}
}

void Blstm::get_results(vector<double> &resultVals) const
{
	resultVals.clear();

	for (unsigned n = 0; n < layers_.back().size() - 1; ++n) {
		resultVals.push_back(layers_.back()[n].get_output_val());
	}
}
