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

New_Blstm::New_Blstm(vector<unsigned> topo, int T, float lR)
{
	_error = 0.0;
	_T = T;
	_bpttTruncate = 4;
	_learningRate = lR;

	_iLSize = topo.at(0);
	_hLSize = topo.at(1);
	_oLSize = topo.at(2);

	_U.clear();
	_U.resize(_hLSize, vector<float> (_iLSize, 0));
	_V.clear();
	_V.resize(_oLSize, vector<float> (_hLSize, 0));
	_W.clear();
	_W.resize(_hLSize, vector<float> (_hLSize, 0));

	//helper::print_matrix("U:", _U);
	//helper::print_matrix("V:", _V);
	//helper::print_matrix("W:", _W);

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
				New_Cell newCell(j, 0, topo.at(i+1), i, false, 1, _T);
				newLayer.add_cell(newCell);
				newLayer.set_type(1);
			} else if (i == (topo.size() - 1))
			{	///	output layer
				New_Cell newCell(j, topo.at(i-1), 0, i, false, 3, _T);
				newLayer.add_cell(newCell);
				newLayer.set_type(3);
			} else 
			{	///	hidden layer
				New_Cell newCell(j, topo.at(i-1), topo.at(i+1), i, true, 2, _T);
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

		int index = thisLayerCells.size();

		New_Cell newCell(index, 0, nextLayerCells.size(), i, false, 4, 1);

		for (int c = 0; c < nextLayerCells.size(); c++) {
			//newCell.create_connection(i, i+1, -1, nextLayerCells.at(c).get_id());
			if (i == 0)
				newCell.set_weight(i, i+1, index, nextLayerCells.at(c).get_id(), 0.35);
			else
				newCell.set_weight(i, i+1, index, nextLayerCells.at(c).get_id(), 0.6);
		}

		layers_.at(i).add_cell(newCell);
		layers_.at(i).set_bias(true);
	}
}

void New_Blstm::add_recursion()
{
	/// recursion only for hidden layers
	for (int i = 1; i < layers_.size() - 1; i++) {
		layers_.at(i).create_recursive_connection(i);
		layers_.at(i).set_recursion(true);
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
				}/*
				if ( (i != (layers_.size() - 1)) && layers_.at(i).is_recursive() )
				{
				///	get weight of the recursive connection
				sum += thisLayerCells.at(c).get_output() * thisLayerCells.at(c).get_weight(
						layers_.at(i).get_id(), layers_.at(i).get_id(),
						thisLayerCells.at(c).get_id(), thisLayerCells.at(c).get_id());
				}
*/
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

float New_Blstm::tanhyp(float x)
{
	float fx = 0.0;
	fx = tanh(x);

	return fx;
}

vector<float> New_Blstm::tanhyp_vec(vector<float> x)
{
	vector<float> fx(x.size(), 0);
	
	for (int i = 0; i < x.size(); i++)
	{
		fx.at(i) = tanh(x.at(i));
	}

	return fx;
}

vector<float> New_Blstm::softmax(vector<float> x)
{
	vector<float> softmax(x.size(), 0);
	float xExpSum = 0;
	
	for (int i = 0; i < x.size(); i++)
	{
		xExpSum += exp(x.at(i));
	}

	for (int i = 0; i < x.size(); i++)
	{
		softmax.at(i) = exp(x.at(i)) / xExpSum;
	}

	return softmax;
}

vector<vector<float>> New_Blstm::matrix_add(vector<vector<float>> A, vector<vector<float>> B)
{
	vector<vector<float>> out(A.size(), vector<float> (A.at(0).size(), 0));

	for (int i = 0; i < A.size(); i++)
	{
		for (int j = 0; j < A.at(0).size(); j++)
		{
			out.at(i).at(j) = A.at(i).at(j) + B.at(i).at(j);
		}
	}

	return out;
}

vector<vector<float>> New_Blstm::matrix_add_with_const(vector<vector<float>> A, vector<vector<float>> B, float x)
{
	vector<vector<float>> out(A.size(), vector<float> (A.at(0).size(), 0));

	for (int i = 0; i < A.size(); i++)
	{
		for (int j = 0; j < A.at(0).size(); j++)
		{
			out.at(i).at(j) = A.at(i).at(j) + x * B.at(i).at(j);
		}
	}

	return out;
}

vector<vector<float>> New_Blstm::matrix_mult(vector<vector<float>> A, vector<vector<float>> B)
{
	if (A.at(0).size() != B.size())
	{
		cout << "Matrices cant be multiplied." << endl;
	}

	vector<vector<float>> out(A.size(), vector<float> (B.at(0).size(), 0));

	for (int i = 0; i < A.size(); i++)
	{
		for (int j = 0; j < B.at(0).size(); j++)
		{
			for (int k = 0; k < B.size(); k++)
			{
				out.at(i).at(j) = A.at(i).at(k) * B.at(k).at(j);
			}
		}
	}

	return out;
}

void New_Blstm::forward_prop(vector<vector<float>> X)
{
	_s.clear();
	_s.resize(_T, vector<float> (_hLSize, 0));
	
	_o.clear();
	_o.resize(_T, vector<float> (_oLSize, 0));

	//helper::print_matrix("U", _U);
	//helper::print_matrix("W", _W);
	//helper::print_matrix("V", _V);

	for (int t = 0; t < _T; t++)
	{
		vector<float> VdotS(_oLSize, 0);
		vector<float> UPlusW(_hLSize, 0);

		for (int iHidden = 0; iHidden < _hLSize; iHidden++)
		{
			float tmp = 0;
			for (int iIn = 0; iIn < _iLSize; iIn++)
			{
				tmp += _U.at(iHidden).at(iIn) * X.at(t).at(iIn);
			}

			float tmp2 = 0;
			for (int iHidden2 = 0; iHidden2 < _hLSize; iHidden2++)
			{
				if (t == 0)
					tmp2 += _W.at(iHidden).at(iHidden2);
				else
					tmp2 += _W.at(iHidden).at(iHidden2) * _s.at(t-1).at(iHidden2);
			}

			UPlusW.at(iHidden) = tanhyp(tmp + tmp2);
		}

		_s.at(t) = UPlusW;

		for (int iOut = 0; iOut < _oLSize; iOut++)
		{
			float tmp = 0;
			for (int iHidden = 0; iHidden < _hLSize; iHidden++)
			{
				tmp += _V.at(iOut).at(iHidden) * _s.at(t).at(iHidden);
			}

			VdotS.at(iOut) = tmp;
		}

		_o.at(t) = softmax( VdotS );
	}

	//helper::print_matrix("_s:", _s);
}

float New_Blstm::calculate_loss(vector<vector<float>> Y)
{
	float L = 0;

	//helper::print_matrix("o", _o);

	for (int t = 0; t < Y.size(); t++) {
		for (int iOut = 0; iOut < Y.at(t).size(); iOut++) {
			L += -1 * Y.at(t).at(iOut) * log(_o.at(t).at(iOut));
			//L += abs(Y.at(t).at(iOut) - _o.at(t).at(iOut));
		}
	}

	L = L / _T;
	return L;
}

vector<vector<float>> New_Blstm::outer(vector<float> a, vector<float> b)
{
	vector<vector<float>> out(a.size(), vector<float> (b.size(), 0));

	for (int i = 0; i < a.size(); i++)
	{
		for (int j = 0; j < b.size(); j++) {
			out.at(i).at(j) = a.at(i) * b.at(j);
		}
	}

	return out;
}

void New_Blstm::bptt(vector<vector<float>> X, vector<vector<float>> Y)
{
	//cout << "-------> bptt()" << endl;
	//helper::print_2matrices_column("_o and Y", _o, Y);

	vector<vector<float>> delta_o(_T, vector<float> (_oLSize, 0));
	
	_dU.clear();
	_dU.resize(_hLSize, vector<float> (_iLSize, 0));
	_dV.clear();
	_dV.resize(_oLSize, vector<float> (_hLSize, 0));
	_dW.clear();
	_dW.resize(_hLSize, vector<float> (_hLSize, 0));



	for (int t = 0; t < delta_o.size(); t++)
	{
		for (int iOut = 0; iOut < delta_o.at(t).size(); iOut++)
		{
			delta_o.at(t).at(iOut) = Y.at(t).at(iOut) - _o.at(t).at(iOut);
		}
	}

	//helper::print_matrix("delta_o:", delta_o);

	/*for (int t = 0; t < Y.size() ; t++)
	{
		Y.at(t) = softmax(Y.at(t));
	}*/

	for (int t = _T-1; t >= 0; t--) {
		vector<vector<float>> dVtmp = outer(delta_o.at(t), _s.at(t));
		_dV = matrix_add(_dV, dVtmp);

		vector<float> delta_t(_hLSize, 0);
		for (int iHidden = 0; iHidden < _hLSize; iHidden++)
		{
			float tmpOut = 0;
			for (int iOut = 0; iOut < _oLSize; iOut++)
			{
				tmpOut += _V.at(iOut).at(iHidden) * delta_o.at(t).at(iOut);
			}
			///	out * ( 1 - tanh^2(s(t))
			delta_t.at(iHidden) = tmpOut * (1 - ( _s.at(t).at(iHidden) * _s.at(t).at(iHidden) ));
		}

		///	bptt
		int bptt_end = t - _bpttTruncate + 1;
		if (bptt_end < 0)
			bptt_end = 0;

		for (int bptt_step = t; bptt_step >= bptt_end; bptt_step--)
		{
			vector<vector<float>> dWtmp( _hLSize, vector<float>(_hLSize, 0) );
			if (bptt_step > 0)
				dWtmp = outer(_s.at(bptt_step - 1), delta_t);
			
			_dW = matrix_add(_dW, dWtmp);

			for (int iHidden = 0; iHidden < _hLSize; iHidden++)
			{
				for (int iIn = 0; iIn < _iLSize; iIn++)
				{
					_dU.at(iHidden).at(iIn) += delta_t.at(iHidden) * X.at(t).at(iIn);
				}
			}

			vector<float> delta_t_new(_hLSize, 0);
			for (int iHidden = 0; iHidden < _hLSize; iHidden++)
			{
				float tmpNewDelta = 0;
				for (int iHidden2 = 0; iHidden2 < _hLSize; iHidden2++)
				{
					tmpNewDelta += _W.at(iHidden).at(iHidden2) * delta_t.at(iHidden);
				}
				if (bptt_step > 0)
				{
					delta_t.at(iHidden) = tmpNewDelta * 
						(1 - ( _s.at(bptt_step - 1).at(iHidden) * _s.at(bptt_step - 1).at(iHidden)));
				} else
					delta_t.at(iHidden) = tmpNewDelta;
			}
		}
	}
	
	float L = calculate_loss(Y);

	L *= _learningRate;

	_U = matrix_add_with_const(_U, _dU, L);
	_W = matrix_add_with_const(_W, _dW, L);
	_V = matrix_add_with_const(_V, _dV, L);

	//helper::print_matrix("_dU:", _dU);
	//helper::print_matrix("_dW:", _dW);
	//helper::print_matrix("_dV:", _dV);
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
	///	print function info
	//cout << "-------> random_weights()" << endl;

	///	get the random generator a seed
	srand(time(0));

	/// iterate the different layers - 1
	/// (we dont need the last layer, because there are no output connections)
	for (int i = 0; i < layers_.size() - 1; i++) {
		///	get the next layer
		/// and pass it to the random_weights() function of this layer
		New_Layer nextLayer = layers_.at(i+1);
		layers_.at(i).random_weights(nextLayer);
		//layers_.at(i).weights(nextLayer);
	}

	for (int m = 0; m < _U.size() ; m++)
	{
		for (int n = 0; n < _U.at(m).size() ; n++)
		{
			float rVal = rand() / float(RAND_MAX);
			rVal = rVal - 0.5;
			rVal = rVal / sqrt(_U.at(m).size());

			_U.at(m).at(n) = rVal;
		}
	}
	
	for (int m = 0; m < _V.size() ; m++)
	{
		for (int n = 0; n < _V.at(m).size() ; n++)
		{
			float rVal = rand() / float(RAND_MAX);
			rVal = rVal - 0.5;
			rVal = rVal / sqrt(_V.at(m).size());

			_V.at(m).at(n) = rVal;
		}
	}
	
	for (int m = 0; m < _W.size() ; m++)
	{
		for (int n = 0; n < _W.at(m).size() ; n++)
		{
			float rVal = rand() / float(RAND_MAX);
			rVal = rVal - 0.5;
			rVal = rVal / sqrt(_W.at(m).size());

			_W.at(m).at(n) = rVal;
		}
	}
	
	//helper::print_matrix("U:", _U);
	//helper::print_matrix("V:", _V);
	//helper::print_matrix("W:", _W);
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

void New_Blstm::set_target_vals(vector<float> targets)
{	
	layers_.back().set_targets(targets);
}

void New_Blstm::print_result(vector<vector<float>> Y)
{	
	helper::print_2matrices_column("_o and Y", _o, Y);
}
